#define _POSIX_C_SOURCE 200809L

#include "gop_nal.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/buffer.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libavutil/pixdesc.h>

typedef struct {
    int64_t pts;
    int64_t packet;
} FrameInfo;

typedef struct {
    int64_t packet;
    int64_t frame;
    int64_t pts;
    int64_t frames;
} GopInfo;

typedef struct {
    int64_t pts;
    int64_t dts;
} PacketTime;

typedef struct {
    int video_stream;
    enum AVCodecID codec_id;
    const char *codec;
    int width;
    int height;
    enum AVPixelFormat pix_fmt;
    AVRational time_base;
    AVRational fps;
    int64_t start_pts;
    int64_t packets;
    FrameInfo *frame;
    size_t frame_count;
    size_t frame_capacity;
    GopInfo *gop;
    size_t gop_count;
    size_t gop_capacity;
    PacketTime *packet_time;
    size_t packet_capacity;
} MediaIndex;

typedef struct {
    int fd;
} OutputIO;

typedef struct {
    AVFormatContext *format;
    AVStream *stream;
    OutputIO io;
    char *path;
    int header_written;
} OutputPart;

static void errorf(const char *format, ...)
{
    va_list args;
    va_start(args, format);
    fprintf(stderr, "aji_gop_media: ");
    vfprintf(stderr, format, args);
    fputc('\n', stderr);
    va_end(args);
}

static const char *av_error(int error, char text[AV_ERROR_MAX_STRING_SIZE])
{
    av_strerror(error, text, AV_ERROR_MAX_STRING_SIZE);
    return text;
}

static void free_index(MediaIndex *index)
{
    free(index->frame);
    free(index->gop);
    free(index->packet_time);
    memset(index, 0, sizeof(*index));
}

static int grow_array(void **array, size_t *capacity, size_t count,
                      size_t element_size)
{
    size_t next;
    void *replacement;

    if (count < *capacity)
        return 0;
    next = *capacity ? *capacity * 2 : 128;
    if (next <= count)
        next = count + 1;
    if (next > SIZE_MAX / element_size)
        return -1;
    replacement = realloc(*array, next * element_size);
    if (!replacement)
        return -1;
    *array = replacement;
    *capacity = next;
    return 0;
}

static int append_frame(MediaIndex *index, const AVFrame *frame)
{
    int64_t packet;

    if (!frame->opaque_ref || frame->opaque_ref->size != sizeof(packet)) {
        errorf("decoder did not preserve packet-to-frame metadata");
        return -1;
    }
    memcpy(&packet, frame->opaque_ref->data, sizeof(packet));
    if (frame->best_effort_timestamp == AV_NOPTS_VALUE) {
        errorf("decoded frame %zu has no presentation timestamp", index->frame_count);
        return -1;
    }
    if (index->frame_count == 0) {
        if (frame->width <= 0 || frame->height <= 0 ||
            frame->format == AV_PIX_FMT_NONE) {
            errorf("first decoded frame has an incomplete format");
            return -1;
        }
        index->width = frame->width;
        index->height = frame->height;
        index->pix_fmt = frame->format;
    } else if (frame->width != index->width || frame->height != index->height ||
               frame->format != index->pix_fmt) {
        errorf("video format changes at decoded frame %zu", index->frame_count);
        return -1;
    }
    if ((frame->flags & AV_FRAME_FLAG_INTERLACED) != 0) {
        errorf("interlaced video is unsupported (frame %zu)", index->frame_count);
        return -1;
    }
    if (grow_array((void **)&index->frame, &index->frame_capacity,
                   index->frame_count, sizeof(*index->frame)) < 0) {
        errorf("out of memory recording decoded frames");
        return -1;
    }
    index->frame[index->frame_count].pts = frame->best_effort_timestamp;
    index->frame[index->frame_count].packet = packet;
    index->frame_count++;
    if (index->frame_count % 1000 == 0)
        fprintf(stderr, "aji_gop_media: scanned %zu decoded frames\n",
                index->frame_count);
    return 0;
}

static int receive_frames(AVCodecContext *decoder, AVFrame *frame,
                          MediaIndex *index)
{
    for (;;) {
        int result = avcodec_receive_frame(decoder, frame);
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF)
            return 0;
        if (result < 0) {
            char text[AV_ERROR_MAX_STRING_SIZE];
            errorf("video decode failed: %s", av_error(result, text));
            return -1;
        }
        if (append_frame(index, frame) < 0)
            return -1;
        av_frame_unref(frame);
    }
}

static unsigned int nal_length_size(const AVCodecParameters *parameters)
{
    const unsigned char *extra = parameters->extradata;
    size_t size = parameters->extradata_size > 0
        ? (size_t)parameters->extradata_size : 0;

    if (parameters->codec_id == AV_CODEC_ID_H264 && size >= 5 && extra[0] == 1)
        return (extra[4] & 3) + 1;
    if (parameters->codec_id == AV_CODEC_ID_HEVC && size >= 22 && extra[0] == 1)
        return (extra[21] & 3) + 1;
    return 0;
}

static int append_gop_candidate(MediaIndex *index, int64_t packet,
                                int64_t pts)
{
    if (grow_array((void **)&index->gop, &index->gop_capacity,
                   index->gop_count, sizeof(*index->gop)) < 0) {
        errorf("out of memory recording GOP boundaries");
        return -1;
    }
    index->gop[index->gop_count].packet = packet;
    index->gop[index->gop_count].frame = -1;
    index->gop[index->gop_count].pts = pts;
    index->gop[index->gop_count].frames = 0;
    index->gop_count++;
    return 0;
}

static int record_packet(MediaIndex *index, const AVPacket *packet,
                         unsigned int length_size)
{
    int codec = index->codec_id == AV_CODEC_ID_H264 ? 264 : 265;
    int kind = aji_gop_packet_kind(packet->data, packet->size, codec, length_size);
    int64_t packet_number = index->packets;

    if (kind < 0) {
        errorf("video packet %" PRId64 " has malformed, unsupported, or ambiguous NAL data",
               packet_number);
        return -1;
    }
    if ((packet->flags & AV_PKT_FLAG_KEY) != 0 && kind != 1) {
        errorf("video packet %" PRId64 " is a non-IDR random-access point",
               packet_number);
        return -1;
    }
    if (grow_array((void **)&index->packet_time, &index->packet_capacity,
                   (size_t)packet_number, sizeof(*index->packet_time)) < 0) {
        errorf("out of memory recording packet timestamps");
        return -1;
    }
    index->packet_time[packet_number].pts = packet->pts;
    index->packet_time[packet_number].dts = packet->dts;
    if (kind == 1 && append_gop_candidate(index, packet_number, packet->pts) < 0)
        return -1;
    index->packets++;
    return 0;
}

static int validate_cfr(MediaIndex *index)
{
    size_t i;
    AVRational frame_time = av_inv_q(index->fps);

    if (index->frame_count == 0) {
        errorf("video decodes to zero frames");
        return -1;
    }
    index->start_pts = index->frame[0].pts;
    for (i = 0; i < index->frame_count; i++) {
        int64_t offset = av_rescale_q_rnd((int64_t)i, frame_time,
            index->time_base, AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX);
        int64_t expected;
        int64_t difference;
        __int128 doubled_error;
        __int128 frame_ticks;

        if ((offset > 0 && index->start_pts > INT64_MAX - offset) ||
            (offset < 0 && index->start_pts < INT64_MIN - offset)) {
            errorf("presentation timestamp grid overflows at frame %zu", i);
            return -1;
        }
        expected = index->start_pts + offset;
        difference = index->frame[i].pts >= expected
            ? index->frame[i].pts - expected : expected - index->frame[i].pts;
        doubled_error = (__int128)difference * 2 * index->fps.num * index->time_base.num;
        frame_ticks = (__int128)index->fps.den * index->time_base.den;
        if (difference > 1 || doubled_error >= frame_ticks) {
            errorf("variable or malformed frame timestamp at frame %zu: got %" PRId64
                   ", expected %" PRId64, i, index->frame[i].pts, expected);
            return -1;
        }
        if (i > 0 && index->frame[i].pts <= index->frame[i - 1].pts) {
            errorf("non-increasing presentation timestamp at frame %zu", i);
            return -1;
        }
    }
    return 0;
}

static int finalize_gops(MediaIndex *index)
{
    size_t g;
    size_t f;

    if (index->gop_count == 0 || index->gop[0].packet != 0) {
        errorf("video does not begin with an independently decodable IDR packet");
        return -1;
    }
    for (g = 0; g < index->gop_count; g++) {
        int found = 0;
        for (f = 0; f < index->frame_count; f++) {
            if (index->frame[f].packet == index->gop[g].packet) {
                if (found) {
                    errorf("IDR packet %" PRId64 " maps to multiple decoded frames",
                           index->gop[g].packet);
                    return -1;
                }
                index->gop[g].frame = (int64_t)f;
                index->gop[g].pts = index->frame[f].pts;
                found = 1;
            }
        }
        if (!found) {
            errorf("IDR packet %" PRId64 " did not produce a decoded frame",
                   index->gop[g].packet);
            return -1;
        }
        if (g > 0 && index->gop[g].frame <= index->gop[g - 1].frame) {
            errorf("IDR display frames are not strictly increasing");
            return -1;
        }
    }

    for (g = 1; g < index->gop_count; g++) {
        int64_t cut_packet = index->gop[g].packet;
        int64_t cut_frame = index->gop[g].frame;
        for (f = 0; f < index->frame_count; f++) {
            if ((index->frame[f].packet < cut_packet && (int64_t)f >= cut_frame) ||
                (index->frame[f].packet >= cut_packet && (int64_t)f < cut_frame)) {
                errorf("packet cut %" PRId64 " crosses display frame order at frame %zu",
                       cut_packet, f);
                return -1;
            }
        }
    }
    for (g = 0; g < index->gop_count; g++) {
        int64_t end = g + 1 < index->gop_count
            ? index->gop[g + 1].frame : (int64_t)index->frame_count;
        index->gop[g].frames = end - index->gop[g].frame;
        if (index->gop[g].frames <= 0) {
            errorf("empty GOP at packet %" PRId64, index->gop[g].packet);
            return -1;
        }
    }
    return 0;
}

static int scan_media(const char *path, MediaIndex *index)
{
    AVFormatContext *input = NULL;
    AVCodecContext *decoder = NULL;
    AVPacket *packet = NULL;
    AVFrame *frame = NULL;
    const AVCodec *codec;
    AVStream *stream;
    unsigned int length_size;
    int result;
    int status = -1;

    memset(index, 0, sizeof(*index));
    result = avformat_open_input(&input, path, NULL, NULL);
    if (result < 0) {
        char text[AV_ERROR_MAX_STRING_SIZE];
        errorf("cannot open '%s': %s", path, av_error(result, text));
        goto done;
    }
    result = avformat_find_stream_info(input, NULL);
    if (result < 0) {
        char text[AV_ERROR_MAX_STRING_SIZE];
        errorf("cannot read stream information: %s", av_error(result, text));
        goto done;
    }
    result = av_find_best_stream(input, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (result < 0) {
        errorf("input has no usable video stream");
        goto done;
    }
    index->video_stream = result;
    stream = input->streams[index->video_stream];
    index->codec_id = stream->codecpar->codec_id;
    if (index->codec_id == AV_CODEC_ID_H264)
        index->codec = "h264";
    else if (index->codec_id == AV_CODEC_ID_HEVC)
        index->codec = "hevc";
    else {
        errorf("unsupported video codec '%s'", avcodec_get_name(index->codec_id));
        goto done;
    }
    index->width = stream->codecpar->width;
    index->height = stream->codecpar->height;
    index->pix_fmt = stream->codecpar->format;
    index->time_base = stream->time_base;
    index->fps = av_guess_frame_rate(input, stream, NULL);
    if (index->time_base.num <= 0 || index->time_base.den <= 0 ||
        index->fps.num <= 0 || index->fps.den <= 0) {
        errorf("video has incomplete time-base or frame-rate metadata");
        goto done;
    }

    codec = avcodec_find_decoder(index->codec_id);
    if (!codec) {
        errorf("software decoder for '%s' is unavailable", index->codec);
        goto done;
    }
    decoder = avcodec_alloc_context3(codec);
    if (!decoder) {
        errorf("cannot allocate decoder");
        goto done;
    }
    result = avcodec_parameters_to_context(decoder, stream->codecpar);
    if (result < 0)
        goto av_failure;
    decoder->flags |= AV_CODEC_FLAG_COPY_OPAQUE;
    result = avcodec_open2(decoder, codec, NULL);
    if (result < 0)
        goto av_failure;
    packet = av_packet_alloc();
    frame = av_frame_alloc();
    if (!packet || !frame) {
        errorf("cannot allocate decode buffers");
        goto done;
    }
    length_size = nal_length_size(stream->codecpar);

    while ((result = av_read_frame(input, packet)) >= 0) {
        if (packet->stream_index == index->video_stream) {
            int64_t packet_number = index->packets;
            if (record_packet(index, packet, length_size) < 0)
                goto done;
            packet->opaque_ref = av_buffer_alloc(sizeof(packet_number));
            if (!packet->opaque_ref) {
                errorf("cannot allocate packet metadata");
                goto done;
            }
            memcpy(packet->opaque_ref->data, &packet_number, sizeof(packet_number));
            result = avcodec_send_packet(decoder, packet);
            if (result == AVERROR(EAGAIN)) {
                if (receive_frames(decoder, frame, index) < 0)
                    goto done;
                result = avcodec_send_packet(decoder, packet);
            }
            if (result < 0)
                goto av_failure;
            if (receive_frames(decoder, frame, index) < 0)
                goto done;
        }
        av_packet_unref(packet);
    }
    if (result != AVERROR_EOF)
        goto av_failure;
    result = avcodec_send_packet(decoder, NULL);
    if (result < 0 && result != AVERROR_EOF)
        goto av_failure;
    if (receive_frames(decoder, frame, index) < 0)
        goto done;
    if (validate_cfr(index) < 0 || finalize_gops(index) < 0)
        goto done;
    fprintf(stderr, "aji_gop_media: scan complete: %zu frames, %" PRId64
            " packets, %zu GOPs\n", index->frame_count, index->packets,
            index->gop_count);
    status = 0;
    goto done;

av_failure:
    {
        char text[AV_ERROR_MAX_STRING_SIZE];
        errorf("FFmpeg operation failed: %s", av_error(result, text));
    }
done:
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&decoder);
    avformat_close_input(&input);
    if (status < 0)
        free_index(index);
    return status;
}

static void print_index(const MediaIndex *index)
{
    const char *pixel_format = av_get_pix_fmt_name(index->pix_fmt);
    size_t i;

    printf("{\"version\":1,\"video_stream\":%d,\"codec\":\"%s\","
           "\"width\":%d,\"height\":%d,\"pix_fmt\":\"%s\","
           "\"time_base\":[%d,%d],\"fps\":[%d,%d],"
           "\"start_pts\":%" PRId64 ",\"frames\":%zu,"
           "\"packets\":%" PRId64 ",\"gops\":[",
           index->video_stream, index->codec, index->width, index->height,
           pixel_format ? pixel_format : "unknown", index->time_base.num,
           index->time_base.den, index->fps.num, index->fps.den,
           index->start_pts, index->frame_count, index->packets);
    for (i = 0; i < index->gop_count; i++) {
        if (i)
            putchar(',');
        printf("{\"packet\":%" PRId64 ",\"frame\":%" PRId64
               ",\"pts\":%" PRId64 ",\"frames\":%" PRId64 "}",
               index->gop[i].packet, index->gop[i].frame,
               index->gop[i].pts, index->gop[i].frames);
    }
    puts("]}");
}

static int parse_cut(const char *text, int64_t *value)
{
    uint64_t parsed = 0;
    const unsigned char *p = (const unsigned char *)text;

    if (!*p)
        return -1;
    while (*p) {
        unsigned int digit;
        if (*p < '0' || *p > '9')
            return -1;
        digit = *p - '0';
        if (parsed > ((uint64_t)INT64_MAX - digit) / 10)
            return -1;
        parsed = parsed * 10 + digit;
        p++;
    }
    *value = (int64_t)parsed;
    return 0;
}

static int validate_cuts(const MediaIndex *index, int argc, char **argv,
                         int64_t **cuts_out)
{
    int64_t *cuts = NULL;
    int i;

    if (argc > 0) {
        cuts = calloc((size_t)argc, sizeof(*cuts));
        if (!cuts) {
            errorf("out of memory parsing cuts");
            return -1;
        }
    }
    for (i = 0; i < argc; i++) {
        size_t g;
        int safe = 0;
        if (parse_cut(argv[i], &cuts[i]) < 0 || cuts[i] <= 0 ||
            cuts[i] >= index->packets || (i > 0 && cuts[i] <= cuts[i - 1])) {
            errorf("invalid decimal packet cut '%s'", argv[i]);
            free(cuts);
            return -1;
        }
        for (g = 1; g < index->gop_count; g++) {
            if (index->gop[g].packet == cuts[i]) {
                safe = 1;
                break;
            }
        }
        if (!safe) {
            errorf("packet cut %" PRId64 " is not a verified IDR boundary", cuts[i]);
            free(cuts);
            return -1;
        }
    }
    *cuts_out = cuts;
    return 0;
}

static int output_write(void *opaque, const uint8_t *buffer, int size)
{
    OutputIO *io = opaque;
    int written = 0;
    while (written < size) {
        ssize_t result = write(io->fd, buffer + written, (size_t)(size - written));
        if (result < 0) {
            if (errno == EINTR)
                continue;
            return AVERROR(errno);
        }
        if (result == 0)
            return AVERROR(EIO);
        written += (int)result;
    }
    return written;
}

static int64_t output_seek(void *opaque, int64_t offset, int whence)
{
    OutputIO *io = opaque;
    if (whence == AVSEEK_SIZE) {
        struct stat status;
        if (fstat(io->fd, &status) < 0)
            return AVERROR(errno);
        return status.st_size;
    }
    whence &= ~AVSEEK_FORCE;
    if (whence != SEEK_SET && whence != SEEK_CUR && whence != SEEK_END)
        return AVERROR(EINVAL);
    {
        off_t result = lseek(io->fd, (off_t)offset, whence);
        return result < 0 ? AVERROR(errno) : result;
    }
}

static void close_output(OutputPart *part, int write_trailer)
{
    if (part->format && write_trailer && part->header_written)
        av_write_trailer(part->format);
    if (part->format && part->format->pb) {
        AVIOContext *avio = part->format->pb;
        avio_flush(avio);
        av_freep(&avio->buffer);
        avio_context_free(&avio);
        part->format->pb = NULL;
    }
    if (part->io.fd >= 0)
        close(part->io.fd);
    avformat_free_context(part->format);
    part->format = NULL;
    part->stream = NULL;
    part->io.fd = -1;
    part->header_written = 0;
}

static int open_output(OutputPart *part, const char *directory, size_t number,
                       const AVStream *input_stream)
{
    size_t path_size = strlen(directory) + 32;
    unsigned char *buffer = NULL;
    int result;

    memset(part, 0, sizeof(*part));
    part->io.fd = -1;
    part->path = malloc(path_size);
    if (!part->path) {
        errorf("out of memory constructing output path");
        return -1;
    }
    snprintf(part->path, path_size, "%s/gop-%06zu.mkv", directory, number);
    part->io.fd = open(part->path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
    if (part->io.fd < 0) {
        errorf("cannot exclusively create '%s': %s", part->path, strerror(errno));
        return -1;
    }
    result = avformat_alloc_output_context2(&part->format, NULL, "matroska", part->path);
    if (result < 0 || !part->format)
        goto failure;
    part->stream = avformat_new_stream(part->format, NULL);
    if (!part->stream)
        goto failure;
    result = avcodec_parameters_copy(part->stream->codecpar, input_stream->codecpar);
    if (result < 0)
        goto failure;
    part->stream->codecpar->codec_tag = 0;
    part->stream->time_base = input_stream->time_base;
    part->stream->avg_frame_rate = input_stream->avg_frame_rate;
    part->stream->r_frame_rate = input_stream->r_frame_rate;
    buffer = av_malloc(32768);
    if (!buffer)
        goto failure;
    part->format->pb = avio_alloc_context(buffer, 32768, 1, &part->io, NULL,
                                          output_write, output_seek);
    if (!part->format->pb)
        goto failure;
    buffer = NULL;
    part->format->pb->seekable = AVIO_SEEKABLE_NORMAL;
    part->format->flags |= AVFMT_FLAG_CUSTOM_IO;
    result = avformat_write_header(part->format, NULL);
    if (result < 0)
        goto failure;
    part->header_written = 1;
    return 0;

failure:
    av_free(buffer);
    if (result < 0) {
        char text[AV_ERROR_MAX_STRING_SIZE];
        errorf("cannot initialize '%s': %s", part->path, av_error(result, text));
    } else {
        errorf("cannot initialize '%s'", part->path);
    }
    close_output(part, 0);
    unlink(part->path);
    return -1;
}

static int64_t segment_shift(const MediaIndex *index, int64_t begin, int64_t end)
{
    int64_t minimum = AV_NOPTS_VALUE;
    int64_t i;
    for (i = begin; i < end; i++) {
        int64_t values[2] = {index->packet_time[i].pts, index->packet_time[i].dts};
        int j;
        for (j = 0; j < 2; j++) {
            if (values[j] != AV_NOPTS_VALUE &&
                (minimum == AV_NOPTS_VALUE || values[j] < minimum))
                minimum = values[j];
        }
    }
    return minimum == AV_NOPTS_VALUE ? 0 : minimum;
}

static int split_media(const char *input_path, const char *directory,
                       int cut_count, char **cut_text)
{
    MediaIndex index;
    AVFormatContext *input = NULL;
    AVPacket *packet = NULL;
    AVStream *input_stream;
    OutputPart part;
    int64_t *cuts = NULL;
    char **created = NULL;
    size_t created_count = 0;
    int64_t packet_number = 0;
    int part_number = 0;
    int result;
    int status = -1;
    struct stat directory_status;

    memset(&index, 0, sizeof(index));
    memset(&part, 0, sizeof(part));
    part.io.fd = -1;
    if (scan_media(input_path, &index) < 0)
        goto done;
    if (validate_cuts(&index, cut_count, cut_text, &cuts) < 0)
        goto done;
    if (stat(directory, &directory_status) < 0 || !S_ISDIR(directory_status.st_mode) ||
        directory_status.st_uid != geteuid()) {
        errorf("output directory must already exist and be owned by the current user: '%s'",
               directory);
        goto done;
    }
    created = calloc((size_t)cut_count + 1, sizeof(*created));
    if (!created) {
        errorf("out of memory recording output files");
        goto done;
    }
    result = avformat_open_input(&input, input_path, NULL, NULL);
    if (result < 0)
        goto av_failure;
    result = avformat_find_stream_info(input, NULL);
    if (result < 0)
        goto av_failure;
    if (index.video_stream >= (int)input->nb_streams) {
        errorf("video stream changed while reopening input");
        goto done;
    }
    input_stream = input->streams[index.video_stream];
    packet = av_packet_alloc();
    if (!packet) {
        errorf("cannot allocate remux packet");
        goto done;
    }
    if (open_output(&part, directory, 0, input_stream) < 0)
        goto done;
    created[created_count++] = part.path;
    part.path = NULL;

    while ((result = av_read_frame(input, packet)) >= 0) {
        if (packet->stream_index == index.video_stream) {
            int64_t end;
            int64_t shift;
            if (part_number < cut_count && packet_number == cuts[part_number]) {
                result = av_write_trailer(part.format);
                if (result < 0)
                    goto av_failure;
                part.header_written = 0;
                close_output(&part, 0);
                part_number++;
                if (open_output(&part, directory, (size_t)part_number, input_stream) < 0)
                    goto done;
                created[created_count++] = part.path;
                part.path = NULL;
            }
            end = part_number < cut_count ? cuts[part_number] : index.packets;
            shift = segment_shift(&index,
                part_number == 0 ? 0 : cuts[part_number - 1], end);
            if (packet->pts != AV_NOPTS_VALUE)
                packet->pts -= shift;
            if (packet->dts != AV_NOPTS_VALUE)
                packet->dts -= shift;
            av_packet_rescale_ts(packet, input_stream->time_base, part.stream->time_base);
            packet->stream_index = 0;
            packet->pos = -1;
            result = av_interleaved_write_frame(part.format, packet);
            if (result < 0)
                goto av_failure;
            packet_number++;
        }
        av_packet_unref(packet);
    }
    if (result != AVERROR_EOF)
        goto av_failure;
    if (packet_number != index.packets) {
        errorf("input changed during split: expected %" PRId64 ", copied %" PRId64
               " video packets", index.packets, packet_number);
        goto done;
    }
    result = av_write_trailer(part.format);
    if (result < 0)
        goto av_failure;
    part.header_written = 0;
    close_output(&part, 0);
    fprintf(stderr, "aji_gop_media: split complete: %d parts, %" PRId64
            " video packets\n", cut_count + 1, packet_number);
    status = 0;
    goto done;

av_failure:
    {
        char text[AV_ERROR_MAX_STRING_SIZE];
        errorf("FFmpeg operation failed during split: %s", av_error(result, text));
    }
done:
    free(part.path);
    close_output(&part, 0);
    av_packet_free(&packet);
    avformat_close_input(&input);
    if (status < 0) {
        size_t i;
        for (i = 0; i < created_count; i++)
            unlink(created[i]);
    }
    if (created) {
        size_t i;
        for (i = 0; i < created_count; i++)
            free(created[i]);
    }
    free(created);
    free(cuts);
    free_index(&index);
    return status;
}

static void usage(FILE *file)
{
    fprintf(file,
        "usage:\n"
        "  aji_gop_media scan INPUT\n"
        "  aji_gop_media split INPUT OUTPUT_DIR [CUT ...]\n"
        "CUT is a verified decimal video-packet index emitted by scan.\n");
}

int main(int argc, char **argv)
{
    if (argc == 3 && strcmp(argv[1], "scan") == 0) {
        MediaIndex index;
        int status = scan_media(argv[2], &index);
        if (status == 0)
            print_index(&index);
        free_index(&index);
        return status == 0 ? 0 : 1;
    }
    if (argc >= 4 && strcmp(argv[1], "split") == 0)
        return split_media(argv[2], argv[3], argc - 4, argv + 4) == 0 ? 0 : 1;
    usage(stderr);
    return 2;
}
