/* GPU regression test for the actual encoder's final-resize path.
 * Build from the project root:
 * cc -O2 -ffunction-sections -fdata-sections -Iinclude \
 *   -I/usr/local/cuda/include sanity/resize_cuda.c -Wl,--gc-sections \
 *   $(pkg-config --cflags --libs libavformat libavcodec libavfilter libavutil libswscale) \
 *   -o /tmp/aji_resize_cuda_test
 * Run: /tmp/aji_resize_cuda_test
 * Requires a CUDA GPU and FFmpeg with scale_cuda. No TensorRT engine needed.
 */
#define main aji_encode_cli_main
#include "../src/encode.c"
#undef main

static int check_resize(enum AVPixelFormat format)
{
    enc_ctx c = {0};
    AVFrame *host = NULL, *input = NULL, *output = NULL;
    int result = 1;
    c.up_w = 7680;
    c.up_h = 4320;
    c.out_w = 3840;
    c.out_h = 2160;
    c.sw_fmt = format;
    c.ifmt = avformat_alloc_context();
    if (!c.ifmt) goto done;
    AVStream *stream = avformat_new_stream(c.ifmt, NULL);
    if (!stream) goto done;
    stream->time_base = (AVRational){1001, 24000};
    if (av_hwdevice_ctx_create(&c.hw_device, AV_HWDEVICE_TYPE_CUDA,
                              NULL, NULL, 0) < 0) goto done;
    c.aji_pool = av_hwframe_ctx_alloc(c.hw_device);
    if (!c.aji_pool) goto done;
    AVHWFramesContext *pool = (AVHWFramesContext *)c.aji_pool->data;
    pool->format = AV_PIX_FMT_CUDA;
    pool->sw_format = format;
    pool->width = c.up_w;
    pool->height = c.up_h;
    if (av_hwframe_ctx_init(c.aji_pool) < 0) goto done;
    host = av_frame_alloc();
    input = av_frame_alloc();
    output = av_frame_alloc();
    if (!host || !input || !output) goto done;
    host->format = format;
    host->width = c.up_w;
    host->height = c.up_h;
    if (av_frame_get_buffer(host, 0) < 0) goto done;
    memset(host->data[0], 0, (size_t)host->linesize[0] * host->height);
    memset(host->data[1], 0, (size_t)host->linesize[1] * host->height / 2);
    if (av_hwframe_get_buffer(c.aji_pool, input, 0) < 0 ||
        av_hwframe_transfer_data(input, host, 0) < 0) goto done;

    for (int i = 0; i < 3; i++) {
        input->pts = i;
        if (init_resize(&c, input) < 0 ||
            av_buffersrc_add_frame_flags(c.buf_src, input,
                                         AV_BUFFERSRC_FLAG_KEEP_REF) < 0 ||
            av_buffersink_get_frame(c.buf_sink, output) < 0) goto done;
        if (output->width != c.out_w || output->height != c.out_h ||
            output->format != AV_PIX_FMT_CUDA || !output->hw_frames_ctx ||
            output->pts != i ||
            ((AVHWFramesContext *)output->hw_frames_ctx->data)->sw_format != format)
            goto done;
        av_frame_unref(output);
    }
    printf("PASS: %s CUDA 7680x4320 -> 3840x2160, 3 frames, PTS preserved\n",
           av_get_pix_fmt_name(format));
    result = 0;
done:
    if (result) fprintf(stderr, "FAIL: %s final CUDA resize\n", av_get_pix_fmt_name(format));
    av_frame_free(&output);
    av_frame_free(&input);
    av_frame_free(&host);
    avfilter_graph_free(&c.graph);
    av_buffer_unref(&c.aji_pool);
    av_buffer_unref(&c.hw_device);
    avformat_free_context(c.ifmt);
    return result;
}

int main(void)
{
    return check_resize(AV_PIX_FMT_NV12) || check_resize(AV_PIX_FMT_P010LE);
}
