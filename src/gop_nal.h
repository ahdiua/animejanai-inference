#ifndef AJI_GOP_NAL_H
#define AJI_GOP_NAL_H

#include <stddef.h>

/* Returns 1 for an IDR VCL packet, 0 for an ordinary VCL packet, and -1
 * for malformed, unsupported, non-IDR random-access, or ambiguous input.
 * nal_length_size is 0 for Annex B or 1, 2, or 4 for length framing. */
int aji_gop_packet_kind(const unsigned char *data, size_t size,
                        int codec, unsigned int nal_length_size);

#endif
