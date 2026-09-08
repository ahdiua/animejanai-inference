#include "gop_nal.h"

#include <assert.h>
#include <stddef.h>

static void test_h264_length_prefixed(void)
{
    const unsigned char idr4[] = {0, 0, 0, 2, 0x65, 0x80};
    const unsigned char ordinary2[] = {0, 2, 0x41, 0x80};
    const unsigned char idr1[] = {2, 0x65, 0x80};
    const unsigned char contradictory[] = {
        0, 0, 0, 2, 0x65, 0x80,
        0, 0, 0, 2, 0x41, 0x80,
    };

    assert(aji_gop_packet_kind(idr4, sizeof idr4, 264, 4) == 1);
    assert(aji_gop_packet_kind(ordinary2, sizeof ordinary2, 264, 2) == 0);
    assert(aji_gop_packet_kind(idr1, sizeof idr1, 264, 1) == 1);
    assert(aji_gop_packet_kind(contradictory, sizeof contradictory, 264, 4) == -1);
}

static void test_hevc_length_prefixed(void)
{
    const unsigned char idr19[] = {0, 0, 0, 3, 19 << 1, 1, 0x80};
    const unsigned char idr20[] = {0, 3, 20 << 1, 1, 0x80};
    const unsigned char ordinary[] = {3, 1 << 1, 1, 0x80};
    const unsigned char cra[] = {0, 0, 0, 3, 21 << 1, 1, 0x80};

    assert(aji_gop_packet_kind(idr19, sizeof idr19, 265, 4) == 1);
    assert(aji_gop_packet_kind(idr20, sizeof idr20, 265, 2) == 1);
    assert(aji_gop_packet_kind(ordinary, sizeof ordinary, 265, 1) == 0);
    assert(aji_gop_packet_kind(cra, sizeof cra, 265, 4) == -1);
}

static void test_annex_b_start_codes(void)
{
    const unsigned char h264_3[] = {0, 0, 1, 0x65, 0x80};
    const unsigned char h264_4[] = {0, 0, 0, 1, 0x41, 0x80};
    const unsigned char hevc_mixed[] = {
        0, 0, 0, 1, 32 << 1, 1, 0x01,
        0, 0, 1, 19 << 1, 1, 0x80,
    };

    assert(aji_gop_packet_kind(h264_3, sizeof h264_3, 264, 0) == 1);
    assert(aji_gop_packet_kind(h264_4, sizeof h264_4, 264, 0) == 0);
    assert(aji_gop_packet_kind(hevc_mixed, sizeof hevc_mixed, 265, 0) == 1);
}

static void test_malformed_packets(void)
{
    const unsigned char truncated[] = {0, 0, 0, 4, 0x65, 0x80};
    const unsigned char zero[] = {0, 0, 0, 0};
    const unsigned char overflow[] = {0xff, 0xff, 0xff, 0xff, 0x65};
    const unsigned char missing_vcl[] = {0, 0, 0, 2, 0x67, 0x80};
    const unsigned char bad_h264_header[] = {0, 0, 0, 2, 0xe5, 0x80};
    const unsigned char bad_hevc_header[] = {0, 0, 0, 3, 19 << 1, 0, 0x80};
    const unsigned char no_start_code[] = {0x65, 0x80};

    assert(aji_gop_packet_kind(truncated, sizeof truncated, 264, 4) == -1);
    assert(aji_gop_packet_kind(zero, sizeof zero, 264, 4) == -1);
    assert(aji_gop_packet_kind(overflow, sizeof overflow, 264, 4) == -1);
    assert(aji_gop_packet_kind(missing_vcl, sizeof missing_vcl, 264, 4) == -1);
    assert(aji_gop_packet_kind(bad_h264_header, sizeof bad_h264_header, 264, 4) == -1);
    assert(aji_gop_packet_kind(bad_hevc_header, sizeof bad_hevc_header, 265, 4) == -1);
    assert(aji_gop_packet_kind(no_start_code, sizeof no_start_code, 264, 0) == -1);
    assert(aji_gop_packet_kind(NULL, 0, 264, 4) == -1);
    assert(aji_gop_packet_kind(no_start_code, sizeof no_start_code, 999, 0) == -1);
    assert(aji_gop_packet_kind(no_start_code, sizeof no_start_code, 264, 3) == -1);
}

int main(void)
{
    test_h264_length_prefixed();
    test_hevc_length_prefixed();
    test_annex_b_start_codes();
    test_malformed_packets();
    return 0;
}
