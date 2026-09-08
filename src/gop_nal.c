#include "gop_nal.h"

#include <stdint.h>

enum {
    AJI_NAL_NONE = 0,
    AJI_NAL_ORDINARY = 1,
    AJI_NAL_IDR = 2,
};

static int classify_nal(const unsigned char *data, size_t size, int codec)
{
    unsigned int type;

    if (codec == 264) {
        if (size < 1 || (data[0] & 0x80) != 0)
            return -1;
        type = data[0] & 0x1f;
        if (type == 5)
            return AJI_NAL_IDR;
        if (type >= 1 && type <= 4)
            return AJI_NAL_ORDINARY;
        return AJI_NAL_NONE;
    }

    if (codec == 265) {
        if (size < 2 || (data[0] & 0x80) != 0 || (data[1] & 0x07) == 0)
            return -1;
        type = (data[0] >> 1) & 0x3f;
        if (type == 19 || type == 20)
            return AJI_NAL_IDR;
        if (type <= 15)
            return AJI_NAL_ORDINARY;
        if (type >= 16 && type <= 31)
            return -1;
        return AJI_NAL_NONE;
    }

    return -1;
}

static int add_kind(int *seen, int kind)
{
    if (kind < 0)
        return -1;
    if (kind == AJI_NAL_NONE)
        return 0;
    if (*seen != AJI_NAL_NONE && *seen != kind)
        return -1;
    *seen = kind;
    return 0;
}

static size_t start_code_size(const unsigned char *data, size_t size)
{
    if (size >= 3 && data[0] == 0 && data[1] == 0 && data[2] == 1)
        return 3;
    if (size >= 4 && data[0] == 0 && data[1] == 0 &&
        data[2] == 0 && data[3] == 1)
        return 4;
    return 0;
}

static int parse_annex_b(const unsigned char *data, size_t size, int codec)
{
    size_t pos = 0;
    int seen = AJI_NAL_NONE;
    int found = 0;

    while (pos < size) {
        size_t prefix = start_code_size(data + pos, size - pos);
        size_t begin;
        size_t end;

        if (prefix == 0)
            return -1;
        found = 1;
        begin = pos + prefix;
        end = begin;
        while (end < size && start_code_size(data + end, size - end) == 0)
            end++;
        if (end == begin)
            return -1;
        if (add_kind(&seen, classify_nal(data + begin, end - begin, codec)) < 0)
            return -1;
        pos = end;
    }

    if (!found || seen == AJI_NAL_NONE)
        return -1;
    return seen == AJI_NAL_IDR ? 1 : 0;
}

static int parse_length_prefixed(const unsigned char *data, size_t size,
                                 int codec, unsigned int length_size)
{
    size_t pos = 0;
    int seen = AJI_NAL_NONE;

    while (pos < size) {
        uint32_t length = 0;
        unsigned int i;

        if (size - pos < length_size)
            return -1;
        for (i = 0; i < length_size; i++)
            length = (length << 8) | data[pos + i];
        pos += length_size;
        if (length == 0 || (uint64_t)length > (uint64_t)(size - pos))
            return -1;
        if (add_kind(&seen, classify_nal(data + pos, length, codec)) < 0)
            return -1;
        pos += length;
    }

    if (seen == AJI_NAL_NONE)
        return -1;
    return seen == AJI_NAL_IDR ? 1 : 0;
}

int aji_gop_packet_kind(const unsigned char *data, size_t size,
                        int codec, unsigned int nal_length_size)
{
    if (!data || size == 0 || (codec != 264 && codec != 265))
        return -1;
    if (nal_length_size == 0)
        return parse_annex_b(data, size, codec);
    if (nal_length_size != 1 && nal_length_size != 2 && nal_length_size != 4)
        return -1;
    return parse_length_prefixed(data, size, codec, nal_length_size);
}
