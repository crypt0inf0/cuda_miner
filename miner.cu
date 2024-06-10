#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include "sha256.cuh"

#define SHA256_BLOCK_SIZE 32

/*********************** FUNCTION DEFINITIONS ***********************/
__device__ void uint32_to_little_endian(uint32_t value, unsigned char *buffer) {
    buffer[0] = value & 0xFF;
    buffer[1] = (value >> 8) & 0xFF;
    buffer[2] = (value >> 16) & 0xFF;
    buffer[3] = (value >> 24) & 0xFF;
}

void hexStringToByteArray(const char *hexstr, unsigned char *output) {
    while (*hexstr && hexstr[1]) {
        sscanf(hexstr, "%2hhx", output++);
        hexstr += 2;
    }
}

unsigned char* hexstr_to_char(const char* hexstr) {
    size_t len = strlen(hexstr);
    size_t final_len = len / 2;
    unsigned char* chars = (unsigned char*)malloc((final_len + 1));
    for(size_t i = 0, j = 0; j < final_len; i += 2, j++)
        chars[j] = (hexstr[i] % 32 + 9) % 25 * 16 + (hexstr[i+1] % 32 + 9) % 25;
    chars[final_len] = '\0';
    return chars;
}

void hexstr_to_intarray(const char* hexstr, uint32_t* outputloc) {
    size_t len = strlen(hexstr);
    size_t intlen = (len + 7) / 8;
    unsigned char* bytes = hexstr_to_char(hexstr);

    for(size_t i = 0; i < intlen; i++) {
        *(outputloc + i) = ((uint32_t)bytes[i * 4])
            + ((uint32_t)bytes[i * 4 + 1] << 8)
            + ((uint32_t)bytes[i * 4 + 2] << 16)
            + ((uint32_t)bytes[i * 4 + 3] << 24);
    }
    free(bytes);
}

uint32_t reverse32(uint32_t value) {
    return (((value & 0x000000FF) << 24) |
            ((value & 0x0000FF00) << 8) |
            ((value & 0x00FF0000) >> 8) |
            ((value & 0xFF000000) >> 24));
}

void print_bytes(const unsigned char *data, size_t dataLen, int format) {
    for(size_t i = 0; i < dataLen; ++i) {
        printf("%02x", data[i]);
        if (format) {
            printf(((i + 1) % 16 == 0) ? "\n" : " ");
        }
    }
    printf("\n");
}

void print_bytes_reversed(const unsigned char *data, size_t dataLen, int format) {
    for(size_t i = dataLen; i > 0; --i) {
        printf("%02x", data[i - 1]);
        if (format) {
            printf(((i - 1) % 16 == 0) ? "\n" : " ");
        }
    }
    printf("\n");
}

void setDifficulty(uint32_t bits, uint32_t *difficulty) {
    for(int i = 0; i < 8; i++)
        difficulty[i] = 0;

    bits = reverse32(bits);

    char exponent = bits & 0xff;
    uint32_t significand = bits >> 8;

    for(int i = 0; i < 3; i++) {
        unsigned char thisvalue = (unsigned char)(significand >> (8 * i));
        int index = 32 - exponent + i;
        difficulty[index / 4] = difficulty[index / 4] |
            ((unsigned int)thisvalue << (8 * (3 - (index % 4))));
    }
}

__device__ void hashBlock(uint32_t nonce, BYTE* blockHeader, uint32_t *result) {
    uint32_to_little_endian(nonce, blockHeader + 76);

    BYTE buf[SHA256_BLOCK_SIZE];
    SHA256_CTX ctx;

    sha256_init(&ctx);
    sha256_update(&ctx, blockHeader, 80);
    sha256_final(&ctx, buf);

    sha256_init(&ctx);
    sha256_update(&ctx, buf, SHA256_BLOCK_SIZE);
    sha256_final(&ctx, buf);

    memcpy(result, buf, SHA256_BLOCK_SIZE);
}

__global__ void mine_kernel(uint32_t nonce_start, BYTE* blockHeader, uint32_t *hash) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = nonce_start + tid;

    hashBlock(nonce, blockHeader, hash);
}

void prepare_blockHeader(BYTE* blockHeader, const char* version, const char* prev_block_hash, const char* merkle_root, const char* time, const char* nbits) {
    hexStringToByteArray(version, blockHeader);
    hexStringToByteArray(prev_block_hash, blockHeader + 4);
    hexStringToByteArray(merkle_root, blockHeader + 36);
    hexStringToByteArray(time, blockHeader + 68);
    hexStringToByteArray(nbits, blockHeader + 72);
}

uint32_t mineBlock(uint32_t noncestart, const char *version, const char *prev_block_hash, const char *merkle_root, const char *time, const char *nbits) {
    BYTE *blockHeader;
    uint32_t *hash;
    cudaMallocManaged(&blockHeader, 80 * sizeof(BYTE));
    cudaMallocManaged(&hash, SHA256_BLOCK_SIZE * sizeof(uint32_t));

    prepare_blockHeader(blockHeader, version, prev_block_hash, merkle_root, time, nbits);

    uint32_t difficulty[8];
    uint32_t bits[1];
    hexstr_to_intarray(nbits, bits);
    setDifficulty(*bits, difficulty);

    clock_t start = clock();

    while (1) {
        int numBlocks = 1;
        int threads_per_block = 1;
        mine_kernel<<<numBlocks, threads_per_block>>>(noncestart, blockHeader, hash);
        cudaDeviceSynchronize();

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
            break;
        }

        for (int i = 0; i < 8; i++) {
            if (hash[7-i] < difficulty[i]) {
                print_bytes_reversed((unsigned char *)hash, 32, 1);
                cudaFree(blockHeader);
                cudaFree(hash);
                return noncestart;
            } else if (hash[7-i] > difficulty[i]) {
                break;
            }
        }

        noncestart += numBlocks * threads_per_block;

        if (((noncestart - numBlocks * threads_per_block) % 500000) == 0) {
            clock_t end = clock();
            double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
            double hashrate = 500000 / time_spent;

            printf("%f hashes per second\n", hashrate);

            start = clock();
        }
    }

    cudaFree(blockHeader);
    cudaFree(hash);

    return 0;
}

int main() {
    const char *version = "01000000";
    const char *prev_block_hash = "0000000000000000000000000000000000000000000000000000000000000000";
    const char *merkle_root = "3BA3EDFD7A7B12B27AC72C3E67768F617FC81BC3888A51323A9FB8AA4B1E5E4A";
    const char *time = "29AB5F49";
    const char *nbits = "FFFF001D";

    uint32_t nonce = mineBlock(2083236893, version, prev_block_hash, merkle_root, time, nbits);
    printf("Nonce found: %u\n", nonce);

    return 0;
}
  