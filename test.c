#include <png.h>
#include "mpi.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int width, height;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers = NULL;

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

    row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_byte*)malloc(png_get_rowbytes(png_ptr, info_ptr));
    }

    png_read_image(png_ptr, row_pointers);

    fclose(fp);
    png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
}

int main(int argc, char *argv[]) {
    int rank, nproc;
    char *file_name;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "USAGE: mpiexec -n <number of cores> ./pngtest filename\n");
        }
        MPI_Abort(comm, 1);
    }

    file_name = argv[1];

    if (rank == 0) {
        read_png(file_name);
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

    // Allocate memory for each process to receive its segment
    png_bytep *segment = (png_bytep*)malloc(sizeof(png_bytep) * (end_row - start_row + 1));
    for (int y = 0; y <= end_row - start_row; y++) {
        segment[y] = (png_byte*)malloc(png_get_rowbytes(width, bit_depth));
    }

    if (rank == 0) {
        // Scatter rows to all processes
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
                MPI_Send(row_pointers[target_start_row + y], png_get_rowbytes(width, bit_depth), MPI_BYTE, i, 0, comm);
            }
        }

        // Copy the rows for rank 0's segment
        for (int y = 0; y <= end_row - start_row; y++) {
            memcpy(segment[y], row_pointers[start_row + y], png_get_rowbytes(width, bit_depth));
        }
    } else {
        // Receive rows for this process's segment
        for (int y = 0; y <= end_row - start_row; y++) {
            MPI_Recv(segment[y], png_get_rowbytes(width, bit_depth), MPI_BYTE, 0, 0, comm, MPI_STATUS_IGNORE);
        }
    }

    // Process the segment with local thresholding (convert to black and white)
    int threshold = 128; // Example threshold value
    for (int y = 0; y <= end_row - start_row; y++) {
        png_bytep row = segment[y];
        for (int x = 0; x < width; x++) {
            png_bytep px = &(row[x * 4]);
            int grayscale_value = (px[0] + px[1] + px[2]) / 3;
            if (grayscale_value > threshold) {
                px[0] = px[1] = px[2] = 255;  // Set to white
            } else {
                px[0] = px[1] = px[2] = 0;    // Set to black
            }
        }
    }

    // Gather the processed segments back to rank 0
    if (rank == 0) {
        for (int y = 0; y <= end_row - start_row; y++) {
            memcpy(row_pointers[start_row + y], segment[y], png_get_rowbytes(width, bit_depth));
        }

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
                MPI_Recv(row_pointers[target_start_row + y], png_get_rowbytes(width, bit_depth), MPI_BYTE, i, 0, comm, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (int y = 0; y <= end_row - start_row; y++) {
            MPI_Send(segment[y], png_get_rowbytes(width, bit_depth), MPI_BYTE, 0, 0, comm);
        }
    }

    if (rank == 0) {
        write_png("output.png");
    }

    for (int y = 0; y <= end_row - start_row; y++) {
        free(segment[y]);
    }
    free(segment);

    if (rank == 0) {
        for (int y = 0; y < height; y++) {
            free(row_pointers[y]);
        }
        free(row_pointers);
    }

    MPI_Finalize();
    return 0;
}
