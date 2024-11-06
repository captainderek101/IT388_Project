/*
 * Proof-of-concept program to demonstrate how
 * an image can be logically "split" using C and MPI
 * for processing in parallel
 * 
 * By Derek Reynolds 10/29/24 
 *
 * Compile:
 *   mpicc -o pngtest libpng_test.c -lpng
 * Execute:
 *   mpiexec -n <number of cores> ./pngtest
*/

#include <png.h>
#include "mpi.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

int width, height;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers = NULL;

int write_png(char *file_name)
{
    png_structp png_ptr;
    png_infop info_ptr;
    FILE *fp;

    if ((fp = fopen(file_name, "wb")) == NULL)
        return (1); // ERROR
    
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (png_ptr == NULL)
    {
        fclose(fp);
        return (1); // ERROR
    }

    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        fclose(fp);
        png_destroy_write_struct(&png_ptr, NULL);
        return (1); // ERROR
    }
    
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        /* Free all of the memory associated with the png_ptr and info_ptr */
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        /* If we get here, we had a problem reading the file */
        return (1); // ERROR
    }
    
    /* Set up the input control if you are using standard C streams */
    png_init_io(png_ptr, fp);

    // // Output is 8bit depth, RGBA format.
    // png_set_IHDR(
    //     png_ptr,
    //     info_ptr,
    //     width, height,
    //     8,
    //     PNG_COLOR_TYPE_RGBA,
    //     PNG_INTERLACE_NONE,
    //     PNG_COMPRESSION_TYPE_DEFAULT,
    //     PNG_FILTER_TYPE_DEFAULT
    // );
    png_write_info(png_ptr, info_ptr);
    
    if (!row_pointers)
        abort();

    /* Write the entire image in one go */
    png_write_image(png_ptr, row_pointers);
    png_write_end(png_ptr, NULL);
    
    for(int y = 0; y < height; y++) {
        free(row_pointers[y]);
    }
    free(row_pointers);
    

    fclose(fp);

    png_destroy_write_struct(&png_ptr, &info_ptr);

    return 0;
}

int read_png(char *file_name)  /* We need to open the file */
{
    png_structp png_ptr;
    png_infop info_ptr;
    FILE *fp;

    if ((fp = fopen(file_name, "rb")) == NULL)
        return (1); // ERROR
    /* Create and initialize the png_struct with the desired error handler
    * functions.  If you want to use the default stderr and longjump method,
    * you can supply NULL for the last three parameters.  We also supply the
    * the compiler header file version, so that we know if the application
    * was compiled with a compatible version of the library.  REQUIRED
    */
    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

    if (png_ptr == NULL)
    {
        fclose(fp);
        return (1); // ERROR
    }

    /* Allocate/initialize the memory for image information.  REQUIRED. */
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL)
    {
        fclose(fp);
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return (1); // ERROR
    }
    
    if (setjmp(png_jmpbuf(png_ptr)))
    {
        /* Free all of the memory associated with the png_ptr and info_ptr */
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        fclose(fp);
        /* If we get here, we had a problem reading the file */
        return (1); // ERROR
    }

    /* Set up the input control if you are using standard C streams */
    png_init_io(png_ptr, fp);


    png_read_info(png_ptr, info_ptr);

    width      = png_get_image_width(png_ptr, info_ptr);
    height     = png_get_image_height(png_ptr, info_ptr);
    color_type = png_get_color_type(png_ptr, info_ptr);
    bit_depth  = png_get_bit_depth(png_ptr, info_ptr);
    
    /* The easiest way to read the image: */
    png_bytep row_pointers[height]; // TODO fix malloc

    /* Clear the pointer array */
    for (int row = 0; row < height; row++)
        row_pointers[row] = NULL;

    for (int row = 0; row < height; row++)
        row_pointers[row] = png_malloc(png_ptr, png_get_rowbytes(png_ptr,
            info_ptr));
    
    printf("ready to read image\n");

    /* Read the entire image in one go */
    png_read_image(png_ptr, row_pointers);

    printf("done reading\n");

    // /* Read rest of file, and get additional chunks in info_ptr - REQUIRED */
    // png_read_end(png_ptr, info_ptr);
    
    /* At this point you have read the entire image */

    /* Close the file */
    fclose(fp);

    /* Clean up after the read, and free any memory allocated - REQUIRED */
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);

    /* That's it */
    return (0); // OK
}

int main(int argc, char* argv[])
{
    /* global variables */
    int rank;
    int nproc;
    char* file_name;
    char buffer[100];
    
    /* Start MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);

    if (rank == 0)
    { 
        if (argc < 2) /* Missing cmd args! */
        {
            printf("---USAGE: mpiexec -n <number of cores> ./pngtest filename\n");
            // MPI_Abort(comm, 1);
        }
        file_name = argv[1];
        printf("hello world\n");
    }

    /* libpng setup */
    read_png(file_name);
    write_png(file_name); // "test.png"

    /* End MPI */
    MPI_Finalize();
    return 0;
}