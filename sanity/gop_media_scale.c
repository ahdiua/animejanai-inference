/* Compile with:
 * cc -std=c11 -O2 -Wall -Wextra -Werror -Isrc sanity/gop_media_scale.c \
 *   src/gop_nal.c $(pkg-config --cflags --libs libavformat libavcodec libavutil) \
 *   -o /tmp/aji-gop-media-scale-test
 */

#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define main aji_gop_media_cli_main
#include "../src/gop_media.c"
#undef main

enum { ITEM_COUNT = 200000 };

static void fill_large_index(MediaIndex *index)
{
    int i;
    memset(index, 0, sizeof(*index));
    index->packets = ITEM_COUNT;
    index->frame_count = ITEM_COUNT;
    index->gop_count = ITEM_COUNT;
    index->packet_time = calloc(ITEM_COUNT, sizeof(*index->packet_time));
    index->frame = calloc(ITEM_COUNT, sizeof(*index->frame));
    index->gop = calloc(ITEM_COUNT, sizeof(*index->gop));
    assert(index->packet_time && index->frame && index->gop);
    for (i = 0; i < ITEM_COUNT; i++) {
        index->frame[i].packet = i;
        index->frame[i].pts = i * 10;
        index->gop[i].packet = i;
    }
}

static void test_large_index_finalizes_in_linear_time(void)
{
    MediaIndex index;
    int i;
    fill_large_index(&index);
    alarm(3);
    assert(finalize_gops(&index) == 0);
    alarm(0);
    for (i = 0; i < ITEM_COUNT; i++) {
        assert(index.gop[i].frame == i);
        assert(index.gop[i].pts == i * 10);
        assert(index.gop[i].frames == 1);
    }
    free_index(&index);
}

static void test_large_sorted_cut_set_validates_in_linear_time(void)
{
    MediaIndex index;
    char **text;
    int64_t *cuts = NULL;
    int i;

    fill_large_index(&index);
    text = calloc(ITEM_COUNT - 1, sizeof(*text));
    assert(text);
    for (i = 1; i < ITEM_COUNT; i++) {
        text[i - 1] = malloc(24);
        assert(text[i - 1]);
        snprintf(text[i - 1], 24, "%d", i);
    }
    alarm(3);
    assert(validate_cuts(&index, ITEM_COUNT - 1, text, &cuts) == 0);
    alarm(0);
    for (i = 1; i < ITEM_COUNT; i++) {
        assert(cuts[i - 1] == i);
        free(text[i - 1]);
    }
    free(cuts);
    free(text);
    free_index(&index);
}

static void test_part_timestamp_shifts_are_computed_once_in_linear_time(void)
{
    MediaIndex index = {0};
    int64_t *cuts;
    int64_t *shifts = NULL;
    int i;

    index.packets = ITEM_COUNT;
    index.packet_time = calloc(ITEM_COUNT, sizeof(*index.packet_time));
    cuts = calloc(ITEM_COUNT - 1, sizeof(*cuts));
    assert(index.packet_time && cuts);
    for (i = 0; i < ITEM_COUNT; i++) {
        index.packet_time[i].pts = i * 10;
        index.packet_time[i].dts = i * 10 - 2;
        if (i > 0)
            cuts[i - 1] = i;
    }
    index.packet_time[0].dts = AV_NOPTS_VALUE;
    alarm(3);
    assert(compute_segment_shifts(&index, cuts, ITEM_COUNT - 1, &shifts) == 0);
    alarm(0);
    assert(shifts[0] == 0);
    for (i = 1; i < ITEM_COUNT; i++)
        assert(shifts[i] == i * 10 - 2);
    free(shifts);
    free(cuts);
    free_index(&index);
}

static void test_display_frames_cannot_cross_packet_intervals(void)
{
    MediaIndex index = {0};
    const int64_t packets[] = {0, 2, 1, 3};
    int i;

    index.packets = 4;
    index.frame_count = 4;
    index.gop_count = 2;
    index.packet_time = calloc(4, sizeof(*index.packet_time));
    index.frame = calloc(4, sizeof(*index.frame));
    index.gop = calloc(2, sizeof(*index.gop));
    assert(index.packet_time && index.frame && index.gop);
    index.gop[0].packet = 0;
    index.gop[1].packet = 2;
    for (i = 0; i < 4; i++) {
        index.frame[i].packet = packets[i];
        index.frame[i].pts = i;
    }
    assert(finalize_gops(&index) < 0);
    free_index(&index);
}

int main(void)
{
    test_large_index_finalizes_in_linear_time();
    test_large_sorted_cut_set_validates_in_linear_time();
    test_part_timestamp_shifts_are_computed_once_in_linear_time();
    test_display_frames_cannot_cross_packet_intervals();
    return 0;
}
