#define _POSIX_C_SOURCE 200809L

/* Compile with:
 * cc -std=c11 -Wall -Wextra -Werror -Isrc sanity/gop_media_io.c \
 *   src/gop_nal.c $(pkg-config --cflags --libs libavformat libavcodec libavutil) \
 *   -o /tmp/aji-gop-media-io-test
 */

#include <assert.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

static int fail_close;

static int injected_close(int fd)
{
    int result = close(fd);
    if (fail_close) {
        errno = EIO;
        return -1;
    }
    return result;
}

#define close injected_close
#define main aji_gop_media_cli_main
#include "../src/gop_media.c"
#undef main
#undef close

static void test_close_output_reports_avio_error(void)
{
    OutputPart part = {0};
    unsigned char *buffer = av_malloc(4096);
    assert(buffer);
    part.io.fd = -1;
    part.format = avformat_alloc_context();
    assert(part.format);
    part.format->pb = avio_alloc_context(buffer, 4096, 1, &part.io,
                                         NULL, output_write, output_seek);
    assert(part.format->pb);
    part.format->pb->error = AVERROR(EIO);
    assert(close_output(&part, 0) < 0);
}

static void test_close_output_reports_close_error(void)
{
    OutputPart part = {0};
    part.io.fd = open("/dev/null", O_WRONLY);
    assert(part.io.fd >= 0);
    fail_close = 1;
    assert(close_output(&part, 0) < 0);
    fail_close = 0;
}

static void assert_directory_empty(const char *path)
{
    DIR *directory = opendir(path);
    struct dirent *entry;
    assert(directory);
    while ((entry = readdir(directory)) != NULL)
        assert(strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0);
    closedir(directory);
}

int main(int argc, char **argv)
{
    if (argc == 3) {
        fail_close = 1;
        assert(split_media(argv[1], argv[2], 0, NULL) < 0);
        fail_close = 0;
        assert_directory_empty(argv[2]);
        return 0;
    }
    assert(argc == 1);
    test_close_output_reports_avio_error();
    test_close_output_reports_close_error();
    return 0;
}
