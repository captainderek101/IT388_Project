/*
 * mpi_threshold.c
 * 
 * Basic implementation of parallel thresholding which was performed by:
 * splitting up an image based off the number of processors so that each one handles that number of pixels 
 * utilizing a threshold value (128) to split the pixels into either black or white
 * and then outputting a finalized black and white thresholded image
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



void apply_sauvola_threshold(png_bytep *segment, int nthreads, int segment_height, int width, int boundary_offset)
{
    double k = 0.5;
    int window_size = 25;
    int half_window = window_size / 2;
    int start_y = boundary_offset;
    int end_y = segment_height - boundary_offset - 1;

    #pragma omp parallel for num_threads(nthreads)
    for (int y = start_y; y <= end_y; ++y) {
        for (int x = half_window; x < width - half_window; ++x) {
            double sum = 0.0, sum_sq = 0.0;
            int count = 0;

            // Compute local mean and stddev within the window
            for (int j = -half_window; j <= half_window; ++j) {
                png_bytep row = segment[y + j]; // Use y+j to access rows above/below
                for (int i = -half_window; i <= half_window; ++i) {
                    png_bytep px;
                    if (color_type == PNG_COLOR_TYPE_RGBA) {
                        px = &(row[(x + i) * 4]);
                    } else {
                        px = &(row[(x + i) * 3]);
                    }
                    int grayscale_value = (px[0] + px[1] + px[2]) / 3;
                    double pixel = grayscale_value / 255.0; // normalize
                    sum += pixel;
                    sum_sq += pixel * pixel;
                    count++;
                }
            }

            double mean = sum / count;
            double variance = (sum_sq / count) - (mean * mean);
            double stddev = sqrt(variance);

            double threshold = mean * (1 + k * ((stddev / 0.5) - 1));

            // Apply threshold to the current pixel
            png_bytep current_px;
            if (color_type == PNG_COLOR_TYPE_RGBA) {
                current_px = &(segment[y][x * 4]);
            } else {
                current_px = &(segment[y][x * 3]);
            }

            int current_val = (current_px[0] + current_px[1] + current_px[2]) / 3;
            if (current_val > threshold * 255.0) {
                current_px[0] = current_px[1] = current_px[2] = 255;
            } else {
                current_px[0] = current_px[1] = current_px[2] = 0;
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

   // Write image data
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

   // Grabs the image properties
   width = png_get_image_width(png_ptr, info_ptr);
   height = png_get_image_height(png_ptr, info_ptr);
   color_type = png_get_color_type(png_ptr, info_ptr);
   bit_depth = png_get_bit_depth(png_ptr, info_ptr);

   // Expand grayscale depth to 8 and pallete images to RGB
   if (color_type == PNG_COLOR_TYPE_PALETTE) {
       png_set_palette_to_rgb(png_ptr);
       color_type = PNG_COLOR_TYPE_RGB;
   }

   // Expand grayscale images to 8 bits
   if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
       png_set_expand_gray_1_2_4_to_8(png_ptr);
   }

   // Add alpha channel if necessary if transparency is used
   if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
       png_set_tRNS_to_alpha(png_ptr);
       color_type = PNG_COLOR_TYPE_RGBA;
   }

   // Update the image structure after transformation
   png_read_update_info(png_ptr, info_ptr);

   // Use malloc to allocate memory to each row
   row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
   for (int y = 0; y < height; y++) {
       row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png_ptr, info_ptr));
   }

   // Read the image into our rows
   png_read_image(png_ptr, row_pointers);

   fclose(fp);
   png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
}

int main(int argc, char *argv[]) {
    int rank, nproc;
    char *input_file_name;
    char *output_file_name;
    int nthreads;
    double startTime, elapsedTime;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);

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

    if (rank == 0) {
        read_png(input_file_name);
        png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        info_ptr = png_create_info_struct(png_ptr);

        elapsedTime = MPI_Wtime() - startTime;
        printf("Time to read image (serial): %f ms\n", elapsedTime * 1000);
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, comm);
    MPI_Bcast(&height, 1, MPI_INT, 0, comm);
    MPI_Bcast(&color_type, 1, MPI_BYTE, 0, comm);
    MPI_Bcast(&bit_depth, 1, MPI_BYTE, 0, comm);

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

    // Window parameters used in threshold
    int window_size = 25;
    int boundary_offset = window_size / 2;

    int local_core_height = end_row - start_row + 1;
    int segment_height = local_core_height + 2 * boundary_offset;

    int row_bytes = (color_type == PNG_COLOR_TYPE_RGBA) ? (width * 4) : (width * 3);

    // Allocate memory for the segment including boundary rows
    png_bytep *segment = (png_bytep*)malloc(sizeof(png_bytep) * segment_height);
    for (int y = 0; y < segment_height; y++) {
        segment[y] = (png_byte*)malloc(row_bytes);
    }

    if (rank == 0) {
        // Send rows including boundaries to other processes
        for (int i = 1; i < nproc; i++) {
            int target_start_row, target_end_row;
            if (i < extra_rows) {
                target_start_row = i * (rows_per_proc + 1);
                target_end_row = target_start_row + rows_per_proc;
            } else {
                target_start_row = i * rows_per_proc + extra_rows;
                target_end_row = target_start_row + rows_per_proc - 1;
            }

            int local_core_h = target_end_row - target_start_row + 1;
            int target_segment_height = local_core_h + 2 * boundary_offset;

            // Determine source rows including boundaries
            int send_start = target_start_row - boundary_offset;
            int send_end = target_end_row + boundary_offset;
            if (send_start < 0) send_start = 0;
            if (send_end >= height) send_end = height - 1;

            // Send the actual rows
            for (int y = send_start; y <= send_end; y++) {
                MPI_Send(row_pointers[y], row_bytes, MPI_BYTE, i, 0, comm);
            }
        }

        // Rank 0 copies its own rows (including boundaries)
        int recv_start = start_row - boundary_offset;
        int recv_end = end_row + boundary_offset;
        if (recv_start < 0) recv_start = 0;
        if (recv_end >= height) recv_end = height - 1;
        int actual_segment_rows = recv_end - recv_start + 1;

        // Fill segment with appropriate boundary rows
        int top_padding = boundary_offset - (start_row - recv_start);
        int bottom_padding = boundary_offset - (recv_end - end_row);

        // Fill top boundary
        for (int y = 0; y < top_padding; y++) {
            memcpy(segment[y], row_pointers[recv_start], row_bytes);
        }

        // Core + middle part
        for (int y = 0; y < actual_segment_rows; y++) {
            memcpy(segment[y + top_padding], row_pointers[recv_start + y], row_bytes);
        }

        // Fill bottom boundary
        for (int y = 0; y < bottom_padding; y++) {
            memcpy(segment[top_padding + actual_segment_rows + y], row_pointers[recv_end], row_bytes);
        }
    } else {
        // Receive rows including boundaries
        int recv_start = start_row - boundary_offset;
        int recv_end = end_row + boundary_offset;
        if (recv_start < 0) recv_start = 0;
        if (recv_end >= height) recv_end = height - 1;
        int actual_segment_rows = recv_end - recv_start + 1;

        png_bytep *temp_rows = (png_bytep*)malloc(sizeof(png_bytep) * actual_segment_rows);
        for (int y = 0; y < actual_segment_rows; y++) {
            temp_rows[y] = (png_byte*)malloc(row_bytes);
            MPI_Recv(temp_rows[y], row_bytes, MPI_BYTE, 0, 0, comm, MPI_STATUS_IGNORE);
        }

        int top_padding = boundary_offset - (start_row - recv_start);
        int bottom_padding = boundary_offset - (recv_end - end_row);

        // Fill top boundary
        for (int y = 0; y < top_padding; y++) {
            memcpy(segment[y], temp_rows[0], row_bytes);
        }

        // Core + middle part
        for (int y = 0; y < actual_segment_rows; y++) {
            memcpy(segment[y + top_padding], temp_rows[y], row_bytes);
        }

        // Fill bottom boundary
        for (int y = 0; y < bottom_padding; y++) {
            memcpy(segment[top_padding + actual_segment_rows + y], temp_rows[actual_segment_rows - 1], row_bytes);
        }

        for (int y = 0; y < actual_segment_rows; y++) free(temp_rows[y]);
        free(temp_rows);
    }

    MPI_Barrier(comm);
    startTime = MPI_Wtime();

    // Process the segment
    apply_sauvola_threshold(segment, nthreads, segment_height, width, boundary_offset);

    MPI_Barrier(comm);
    elapsedTime = MPI_Wtime() - startTime;

    if (rank == 0) {
        printf("Time to threshold image (parallel): %f ms\n", elapsedTime * 1000);
        // Copy back only the core portion from rank 0
        for (int y = 0; y < (end_row - start_row + 1); y++) {
            memcpy(row_pointers[start_row + y], segment[y + boundary_offset], row_bytes);
        }

        // Receive core portions from other processes
        for (int i = 1; i < nproc; i++) {
            int target_start_row, target_end_row;
            if (i < extra_rows) {
                target_start_row = i * (rows_per_proc + 1);
                target_end_row = target_start_row + rows_per_proc;
            } else {
                target_start_row = i * rows_per_proc + extra_rows;
                target_end_row = target_start_row + rows_per_proc - 1;
            }

            int core_height = target_end_row - target_start_row + 1;
            for (int y = 0; y < core_height; y++) {
                MPI_Recv(row_pointers[target_start_row + y], row_bytes, MPI_BYTE, i, 0, comm, MPI_STATUS_IGNORE);
            }
        }
    } else {
        // Send back only the core portion
        for (int y = 0; y < (end_row - start_row + 1); y++) {
            MPI_Send(segment[y + boundary_offset], row_bytes, MPI_BYTE, 0, 0, comm);
        }
    }

    if (rank == 0) {
        write_png(output_file_name);
    }

    MPI_Finalize();
    return 0;
}
