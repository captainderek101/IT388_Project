/*
 * mpi_threshold.c
 * 
 * Basic implementation of parallel thresholding which was performed by:
 * splitting up an image based off the number of processors so that each one handles that number of pixels 
 * utilizing a threshold value (128) to split the pixels into either black or white
 * and then outputting a finalized black and white thresholded image
 * Compile: mpicc -o pngThreshold mpi_threshold_omp.c -lpng -fopenmp
 * 
 * Execute: mpiexec -n <numcores> ./pngThreshold <numthreads> <image.png> <outimage.png>
 */

#include <png.h>
#include "mpi.h"
#include "omp.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int width, height;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers = NULL;

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

   // Manager core reads the input png
   if (rank == 0) {
       read_png(input_file_name);
       png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
       info_ptr = png_create_info_struct(png_ptr);
   }

   // Broadcast the image to all processes
   MPI_Bcast(&width, 1, MPI_INT, 0, comm);
   MPI_Bcast(&height, 1, MPI_INT, 0, comm);
   MPI_Bcast(&color_type, 1, MPI_BYTE, 0, comm);
   MPI_Bcast(&bit_depth, 1, MPI_BYTE, 0, comm);

   // Calculate numrows for each processor
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

   // Allocate memory for each process
   int row_bytes;
   if (color_type == PNG_COLOR_TYPE_RGBA) {
       row_bytes = width * 4;
   } else {
       row_bytes = width * 3;
   }
   
   png_bytep *segment = (png_bytep*)malloc(sizeof(png_bytep) * (end_row - start_row + 1));
   for (int y = 0; y <= end_row - start_row; y++) {
       segment[y] = (png_byte*)malloc(row_bytes);
   }

   if (rank == 0) {
       // Scatter the rows to all processers
       for (int i = 1; i < nproc; i++) {
           int target_start_row, target_end_row;
           if (i < extra_rows) {
               target_start_row = i * (rows_per_proc + 1);
               target_end_row = target_start_row + rows_per_proc;
           } else {
               target_start_row = i * rows_per_proc + extra_rows;
               target_end_row = target_start_row + rows_per_proc - 1;
           }

           int num_rows = target_end_row - target_start_row + 1;
           for (int y = 0; y < num_rows; y++) {
               MPI_Send(row_pointers[target_start_row + y], row_bytes, MPI_BYTE, i, 0, comm);
           }
       }

       // Copy the rows for rank 0's portion
       for (int y = 0; y <= end_row - start_row; y++) {
           memcpy(segment[y], row_pointers[start_row + y], row_bytes);
       }
   } else {
       // Receive rows
       for (int y = 0; y <= end_row - start_row; y++) {
           MPI_Recv(segment[y], row_bytes, MPI_BYTE, 0, 0, comm, MPI_STATUS_IGNORE);
       }
   }

   // Process the segment with local thresholding black and white conversion
   int threshold = 128;
#pragma omp parallel for num_threads(nthreads)
   for (int y = 0; y <= end_row - start_row; y++) {
       png_bytep row = segment[y];
       for (int x = 0; x < width; x++) {
           png_bytep px;
           if (color_type == PNG_COLOR_TYPE_RGBA) {
               px = &(row[x * 4]);
           } else {
               px = &(row[x * 3]);
           }
           // Calculate grayscale value
           int grayscale_value = (px[0] + px[1] + px[2]) / 3;
           // Apply threshold to convert to black or white
           if (grayscale_value > threshold) {
               px[0] = px[1] = px[2] = 255;  // Set color to white
           } else {
               px[0] = px[1] = px[2] = 0;    // Set color to black
           }
       }
   }

   // Gather the processed segments
   if (rank == 0) {
       // Copy core 0s segment back to the main image
       for (int y = 0; y <= end_row - start_row; y++) {
           memcpy(row_pointers[start_row + y], segment[y], row_bytes);
       }

       // Receive segments from other processes and assemble the final image
       for (int i = 1; i < nproc; i++) {
           int target_start_row, target_end_row;
           if (i < extra_rows) {
               target_start_row = i * (rows_per_proc + 1);
               target_end_row = target_start_row + rows_per_proc;
           } else {
               target_start_row = i * rows_per_proc + extra_rows;
               target_end_row = target_start_row + rows_per_proc - 1;
           }

           int num_rows = target_end_row - target_start_row + 1;
           for (int y = 0; y < num_rows; y++) {
               MPI_Recv(row_pointers[target_start_row + y], row_bytes, MPI_BYTE, i, 0, comm, MPI_STATUS_IGNORE);
           }
       }
   } else {
       // Send the processed segment back to manager core
       for (int y = 0; y <= end_row - start_row; y++) {
           MPI_Send(segment[y], row_bytes, MPI_BYTE, 0, 0, comm);
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
