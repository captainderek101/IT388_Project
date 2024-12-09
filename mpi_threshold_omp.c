/*
 * mpi_threshold_omp.c
 *
 * Implementation of local thresholding using sauvolas method for thresholding
 * which is performed by splitting up an image file into however many threads there are
 * and then sending out each segment to its thread using MPI send and recieve 
 * it then applying a threshold based off the standard deviation and mean to turn each pixel black or white
 * finally it outputs a finalized image thats completely seperated by black/white
 *
 * Coded by Brooke Bastion, Clay Remen, and Derek Reynolds
 * Written for IT 388 Final Project
 * Compile: mpicc -o pngThreshold mpi_threshold_omp.c -lpng -fopenmp -lm
 * 
 * Execute: mpiexec -n <numcores> ./pngThreshold <numthreads> <image.png> <outimage.png>
 */

#include <math.h>
#include "mpi.h"
#include "omp.h"
#include <png.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int width, height;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers = NULL;

void apply_sauvola_threshold(png_bytep *segment, int nthreads, int window_size, int seg_height, int width)
{
    double k = 0.5;
    int half_window = window_size / 2;

    #pragma omp parallel for num_threads(nthreads)
    for (int y = half_window; y < seg_height - half_window; ++y) {
        png_bytep row = segment[y];
        for (int x = half_window; x < width - half_window; ++x) {
            // Calculate mean and standard deviation in the window
            double sum = 0, sum_sq = 0;
            int count = 0;
            for (int j = -half_window; j <= half_window; ++j) {
                png_bytep row_inner = segment[y + j];
                for (int i = -half_window; i <= half_window; ++i) {
                    png_bytep px;
                    if (color_type == PNG_COLOR_TYPE_RGBA) {
                        px = &(row_inner[(x + i) * 4]);
                    } else {
                        px = &(row_inner[(x + i) * 3]);
                    }
                    int grayscale_value = (px[0] + px[1] + px[2]) / 3;
                    double pixel = grayscale_value / 255.0; // Normalize to [0,1]
                    sum += pixel;
                    sum_sq += pixel * pixel;
                    count++;
                }
            }
            double mean = sum / count;
            double variance = (sum_sq / count) - (mean * mean);
            double stddev = sqrt(variance);
            // Sauvola threshold calculation
            double threshold = mean * (1 + k * ((stddev / 1) - 1));

            // Apply threshold
            png_bytep px;
            if (color_type == PNG_COLOR_TYPE_RGBA) {
                px = &(row[x * 4]);
            } else {
                px = &(row[x * 3]);
            }

            int grayscale_value = (px[0] + px[1] + px[2]) / 3;
            double pixelNorm = grayscale_value / 255.0;
            if (pixelNorm > threshold) {
                px[0] = px[1] = px[2] = 255;  // white
            } else {
                px[0] = px[1] = px[2] = 0;    // black
            }
        }
    }
}

// Writes a png to an image file
void write_png(const char *file_name) {
    FILE *fp = fopen(file_name, "wb");
    if (!fp) {
        fprintf(stderr, "Could not open file %s for writing\n", file_name);
        abort();
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "Could not allocate write struct\n");
        abort();
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "Could not allocate info struct\n");
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        abort();
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error during png creation\n");
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        abort();
    }

    png_init_io(png_ptr, fp);

    png_set_IHDR(
        png_ptr,
        info_ptr,
        width, height,
        bit_depth,
        color_type,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );

    png_write_info(png_ptr, info_ptr);
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);

    fclose(fp);

    png_destroy_write_struct(&png_ptr, &info_ptr);
}

// Reads a png from a image file
void read_png(const char *file_name) {
    FILE *fp = fopen(file_name, "rb");
    if (!fp) {
        fprintf(stderr, "Could not open file %s for reading\n", file_name);
        abort();
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fprintf(stderr, "Could not allocate read struct\n");
        abort();
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        fprintf(stderr, "Could not allocate info struct\n");
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
        abort();
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        fprintf(stderr, "Error during init_io\n");
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
        fclose(fp);
        abort();
    }

    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    width = png_get_image_width(png_ptr, info_ptr);
    height = png_get_image_height(png_ptr, info_ptr);
    color_type = png_get_color_type(png_ptr, info_ptr);
    bit_depth = png_get_bit_depth(png_ptr, info_ptr);

    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
        color_type = PNG_COLOR_TYPE_RGB;
    }

    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    }

    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png_ptr);
        color_type = PNG_COLOR_TYPE_RGBA;
    }

    png_read_update_info(png_ptr, info_ptr);

    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png_ptr, info_ptr));
    }

    png_read_image(png_ptr, row_pointers);

    fclose(fp);
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
}
// Main function to input and threshold the image
int main(int argc, char *argv[]) {
    int rank, nproc;
    char *input_file_name;
    char *output_file_name;
    int nthreads;
    double startTime, elapsedTime;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);

    // Check for the desired number of arguments
    if (argc < 4) {
        if (rank == 0) {
            fprintf(stderr, "USAGE: mpiexec -n <number of cores> ./pngtest <numthreads> <input_filename> <output_filename>\n");
        }
        MPI_Abort(comm, 1);
    }

    nthreads = atoi(argv[1]);
    input_file_name = argv[2];
    output_file_name = argv[3];

    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;

    MPI_Barrier(comm);
    startTime = MPI_Wtime();

    // Manager core reads the input png
    if (rank == 0) {
        read_png(input_file_name);
        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        info_ptr = png_create_info_struct(png_ptr);
        elapsedTime = MPI_Wtime() - startTime;
        printf("Time to read image (serial): %f ms\n", elapsedTime * 1000);
    }

    // Broadcast image properties to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, comm);
    MPI_Bcast(&height, 1, MPI_INT, 0, comm);
    MPI_Bcast(&color_type, 1, MPI_BYTE, 0, comm);
    MPI_Bcast(&bit_depth, 1, MPI_BYTE, 0, comm);

    int window_size = 25;
    int half_window = window_size / 2;

    int rows_per_proc = height / nproc;
    int extra_rows = height % nproc;
    int start_row, end_row;

    if (rank < extra_rows) {
        start_row = rank * (rows_per_proc + 1);
        end_row = start_row + rows_per_proc;
    } else {
        start_row = rank * rows_per_proc + extra_rows;
        end_row = start_row + rows_per_proc - 1;
    }

    // Determine padded start and end rows for boundaries 
    int padded_start = start_row - half_window;
    if (padded_start < 0) {
        padded_start = 0;
    }
    int padded_end = end_row + half_window;
    if (padded_end > height - 1) {
        padded_end = height - 1;
    }

    int padded_height = padded_end - padded_start + 1;
    
    int row_bytes = (color_type == PNG_COLOR_TYPE_RGBA) ? width * 4 : width * 3;

    png_bytep *segment = (png_bytep*)malloc(sizeof(png_bytep) * padded_height);
    for (int y = 0; y < padded_height; y++) {
        segment[y] = (png_byte*)malloc(row_bytes);
    }

    // Scatter rows with boundaries
    if (rank == 0) {
        for (int i = 1; i < nproc; i++) {
            int i_start_row, i_end_row;
            if (i < extra_rows) {
                i_start_row = i * (rows_per_proc + 1);
                i_end_row = i_start_row + rows_per_proc;
            } else {
                i_start_row = i * rows_per_proc + extra_rows;
                i_end_row = i_start_row + rows_per_proc - 1;
            }

            int i_padded_start = i_start_row - half_window;
            if (i_padded_start < 0) {
                i_padded_start = 0;
            }
            int i_padded_end = i_end_row + half_window;
            if (i_padded_end > height - 1) {
                i_padded_end = height - 1;
            }
            int i_padded_height = i_padded_end - i_padded_start + 1;

            for (int y = 0; y < i_padded_height; y++) {
                MPI_Send(row_pointers[i_padded_start + y], row_bytes, MPI_BYTE, i, 0, comm);
            }
        }

        for (int y = 0; y < padded_height; y++) {
            memcpy(segment[y], row_pointers[padded_start + y], row_bytes);
        }
    } else {
        for (int y = 0; y < padded_height; y++) {
            MPI_Recv(segment[y], row_bytes, MPI_BYTE, 0, 0, comm, MPI_STATUS_IGNORE);
        }
    }

    MPI_Barrier(comm);
    startTime = MPI_Wtime();

    // Apply threshold to main segments
    apply_sauvola_threshold(segment, nthreads, window_size, padded_height, width);

    MPI_Barrier(comm);
    elapsedTime = MPI_Wtime() - startTime;

    if (rank == 0) {
        printf("Time to threshold image (parallel): %f ms\n", elapsedTime * 1000);

        // Copy back the segments without the boundaries
        int core_offset = start_row - padded_start;
        int core_height = end_row - start_row + 1;
        for (int y = 0; y < core_height; y++) {
            memcpy(row_pointers[start_row + y], segment[core_offset + y], row_bytes);
        }

        // Receive segments back
        for (int i = 1; i < nproc; i++) {
            int i_start_row, i_end_row;
            if (i < extra_rows) {
                i_start_row = i * (rows_per_proc + 1);
                i_end_row = i_start_row + rows_per_proc;
            } else {
                i_start_row = i * rows_per_proc + extra_rows;
                i_end_row = i_start_row + rows_per_proc - 1;
            }

            int i_core_height = i_end_row - i_start_row + 1;

            for (int y = 0; y < i_core_height; y++) {
                MPI_Recv(row_pointers[i_start_row + y], row_bytes, MPI_BYTE, i, 0, comm, MPI_STATUS_IGNORE);
            }
        }
    } else {
        // Send back only the original portion with no boundaries
        int core_offset = start_row - padded_start;
        int core_height = end_row - start_row + 1;
        for (int y = 0; y < core_height; y++) {
            MPI_Send(segment[core_offset + y], row_bytes, MPI_BYTE, 0, 0, comm);
        }
    }
    
    // Manager writes the final processed image to the output file
    if (rank == 0) {
        write_png(output_file_name);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
