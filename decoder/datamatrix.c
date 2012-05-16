
/******************************************************************************
 ******************************************************************************/

/** \file stdform.c
 *  description here
 *
 * $Id: datamatrix.c 12 2007-06-28 23:20:46Z grizz $
 */

/******************************************************************************
 * L I C E N S E **************************************************************
 ******************************************************************************/

/*
 * Copyright (c) 2007 dev/IT - http://www.devit.com
 *
 * This file is part of libdatamatrix - http://oss.devit.com/datamatrix/
 *
 * libdatamatrix is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * libdatamatrix is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libdatamatrix; if not, write to the Free Software 
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

/******************************************************************************
 * I N C L U D E S ************************************************************
 ******************************************************************************/

#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/******************************************************************************
 * D E F I N E S **************************************************************
 ******************************************************************************/

#define _OFF(w, x, y) ((y) * (w) + (x))

/* reed solomon defines for 8 bit symbols */
#define RS_SYMSZ 8
#define RS_N (1 << RS_SYMSZ) - 1

/******************************************************************************
 * T Y P E D E F S ************************************************************
 ******************************************************************************/

typedef struct
{
   int w, h;
   int dcw, rscw;
} dm_szdef_t;

typedef struct
{
   int sz, idx;
   uint8_t buf[1];
} dm_buf_t;

typedef struct
{
   int w, h;
   uint8_t grid[1];
} dm_region_t;

/******************************************************************************
 * M A C R O S ****************************************************************
 ******************************************************************************/

/******************************************************************************
 * G L O B A L S **************************************************************
 ******************************************************************************/

static dm_szdef_t dm_szdef[] =
   {
      {8, 8, 3, 5},
      {10, 10, 5, 7},
      {12, 12, 8, 10},
      {14, 14, 12, 12},
      {16, 16, 18, 14},
      {18, 18, 22, 18},
      {20, 20, 30, 20},
      {22, 22, 36, 24},
      {24, 24, 44, 28},
      {0, 0, 0, 0}
   };

static void *rs_decoder = NULL; // Reed-Solomon decoder
static const dm_szdef_t *current_size = NULL; // what rs_decoder is initialized for

/******************************************************************************
 * P R O T O T Y P E S ********************************************************
 ******************************************************************************/

/* high level functions */
static char* dm_region_read(dm_region_t *reg, const dm_szdef_t *def, void *rs);
static dm_region_t* dm_region_new(int w, int h);
static char* dm_decode(dm_buf_t *data);

// from fec.h
/* General purpose RS codec, 8-bit symbols */
int decode_rs_char(void *rs,unsigned char *data,int *eras_pos, int no_eras);
void *init_rs_char(int symsize,int gfpoly, int fcr,int prim,int nroots, int pad);
void free_rs_char(void *rs);

/******************************************************************************
 * F U N C T I O N S **********************************************************
 ******************************************************************************/

/******************************************************************************/
/** get a region size definition by width and height
 */

static const dm_szdef_t* dm_szdef_wh(int w, int h)
{
   dm_szdef_t *def = dm_szdef;

   while(def->w)
   {
      if(def->w == w && def->h == h)
         return(def);
      def++;
   }
   return NULL;
}

/******************************************************************************/
/** create a new dm_buf
 */

static dm_buf_t* dm_buf_new(int sz)
{
   dm_buf_t *new;

   if(!(new = calloc(1, sizeof(dm_buf_t) + sz)))
      return (dm_buf_t *)PyErr_NoMemory();
   new->sz = sz;
   return new;
}

/******************************************************************************/
/** get offset of coords, wrap if necessary
 */

static inline unsigned int _woff(dm_region_t *grid, int x, int y)
{
   /* handle wrapping */
   if(x < 0)
   {
      x += grid->w;
      y += 4 - ((grid->w + 4) % 8);
   }
   if(y < 0)
   {
      y += grid->h;
      x += 4 - ((grid->h + 4) % 8);
   }

   return(y * grid->w + x);
}

/******************************************************************************/
/** get a codeword from the matrix at coords x,y
 */

static void _get0(dm_region_t *reg, int x, int y, uint8_t *cw)
{
   uint8_t *grid = reg->grid;


   /* bottom row */
   *cw = grid[_woff(reg, x, y)];
   *cw |= grid[_woff(reg, x-1, y)] << 1;
   *cw |= grid[_woff(reg, x-2, y)] << 2;

   /* next row up */
   *cw |= grid[_woff(reg, x, --y)] << 3;
   *cw |= grid[_woff(reg, x-1, y)] << 4;
   *cw |= grid[_woff(reg, x-2, y)] << 5;

   /* top row */
   *cw |= grid[_woff(reg, x-1, --y)] << 6;
   *cw |= grid[_woff(reg, x-2, y)] << 7;
}

/******************************************************************************/
/** put special case 1
 * @param x width of region
 * @param y height of region
 */

static void _put1(dm_region_t *reg, int x, int y, uint8_t *word)
{
   int w = reg->w;
   uint8_t *grid = reg->grid, cw = *word;


   grid[_OFF(w, --x, 3)] = cw & 0x01;
   grid[_OFF(w, x, 2)] = (cw >> 1) & 0x01;
   grid[_OFF(w, x, 1)] = (cw >> 2) & 0x01;
   grid[_OFF(w, x, 0)] = (cw >> 3) & 0x01;
   grid[_OFF(w, x-1, 0)] = (cw >> 4) & 0x01;
   grid[_OFF(w, 2, --y)] = (cw >> 5) & 0x01;
   grid[_OFF(w, 1, y)] = (cw >> 6) & 0x01;
   grid[_OFF(w, 0, y)] = (cw >> 7) & 0x01;
}

/******************************************************************************/
/** get special case 1
 * @param x width of region
 * @param y height of region
 */

static void _get1(dm_region_t *reg, int x, int y, uint8_t *cw)
{
   int w = reg->w;
   uint8_t *grid = reg->grid;


   *cw = grid[_OFF(w, --x, 3)];
   *cw |= grid[_OFF(w, x, 2)] << 1;
   *cw |= grid[_OFF(w, x, 1)] << 2;
   *cw |= grid[_OFF(w, x, 0)] << 3;
   *cw |= grid[_OFF(w, x-1, 0)] << 4;
   *cw |= grid[_OFF(w, 2, --y)] << 5;
   *cw |= grid[_OFF(w, 1, y)] << 6;
   *cw |= grid[_OFF(w, 0, y)] << 7;
}

/******************************************************************************/
/** put special case 2
 * @param x width of region
 * @param y height of region
 */

static void _put2(dm_region_t *reg, int x, int y, uint8_t *word)
{
   int w = reg->w;
   uint8_t *grid = reg->grid, cw = *word;


   grid[_OFF(w, --x, 1)] = cw & 0x01;
   grid[_OFF(w, x, 0)] = (cw >> 1) & 0x01;
   grid[_OFF(w, x-1, 0)] = (cw >> 2) & 0x01;
   grid[_OFF(w, x-2, 0)] = (cw >> 3) & 0x01;
   grid[_OFF(w, x-3, 0)] = (cw >> 4) & 0x01;
   grid[_OFF(w, 0, y-1)] = (cw >> 5) & 0x01;
   grid[_OFF(w, 0, y-2)] = (cw >> 6) & 0x01;
   grid[_OFF(w, 0, y-3)] = (cw >> 7) & 0x01;
}

/******************************************************************************/
/** get special case 2
 * @param x width of region
 * @param y height of region
 */

static void _get2(dm_region_t *reg, int x, int y, uint8_t *cw)
{
   int w = reg->w;
   uint8_t *grid = reg->grid;


   *cw = grid[_OFF(w, --x, 1)] & 0x01;
   *cw |= grid[_OFF(w, x, 0)] << 1;
   *cw |= grid[_OFF(w, x-1, 0)] << 2;
   *cw |= grid[_OFF(w, x-2, 0)] << 3;
   *cw |= grid[_OFF(w, x-3, 0)] << 4;
   *cw |= grid[_OFF(w, 0, y-1)] << 5;
   *cw |= grid[_OFF(w, 0, y-2)] << 6;
   *cw |= grid[_OFF(w, 0, y-3)] << 7;
}

/******************************************************************************/
/** put special case 3
 * @param x width of region
 * @param y height of region
 */

static void _put3(dm_region_t *reg, int x, int y, uint8_t *word)
{
   int w = reg->w;
   uint8_t *grid = reg->grid, cw = *word;


   grid[_OFF(w, --x, 3)] = cw & 0x01;
   grid[_OFF(w, x, 2)] = (cw >> 1) & 0x01;
   grid[_OFF(w, x, 1)] = (cw >> 2) & 0x01;
   grid[_OFF(w, x, 0)] = (cw >> 3) & 0x01;
   grid[_OFF(w, x-1, 0)] = (cw >> 4) & 0x01;
   grid[_OFF(w, 0, y-1)] = (cw >> 5) & 0x01;
   grid[_OFF(w, 0, y-2)] = (cw >> 6) & 0x01;
   grid[_OFF(w, 0, y-3)] = (cw >> 7) & 0x01;
}

/******************************************************************************/
/** get special case 3
 * @param x width of region
 * @param y height of region
 */

static void _get3(dm_region_t *reg, int x, int y, uint8_t *cw)
{
   int w = reg->w;
   uint8_t *grid = reg->grid;


   *cw = grid[_OFF(w, --x, 3)];
   *cw |= grid[_OFF(w, x, 2)] << 1;
   *cw |= grid[_OFF(w, x, 1)] << 2;
   *cw |= grid[_OFF(w, x, 0)] << 3;
   *cw |= grid[_OFF(w, x-1, 0)] << 4;
   *cw |= grid[_OFF(w, 0, y-1)] << 5;
   *cw |= grid[_OFF(w, 0, y-2)] << 6;
   *cw |= grid[_OFF(w, 0, y-3)] << 7;
}

/******************************************************************************/
/** put special case 4
 * @param x width of region
 * @param y height of region
 */

static void _put4(dm_region_t *reg, int x, int y, uint8_t *word)
{
   int w = reg->w;
   uint8_t *grid = reg->grid, cw = *word;


   grid[_OFF(w, x-1, 1)] = cw & 0x01;
   grid[_OFF(w, x-2, 1)] = (cw >> 1) & 0x01;
   grid[_OFF(w, x-3, 1)] = (cw >> 2) & 0x01;
   grid[_OFF(w, x-1, 0)] = (cw >> 3) & 0x01;
   grid[_OFF(w, x-2, 0)] = (cw >> 4) & 0x01;
   grid[_OFF(w, x-3, 0)] = (cw >> 5) & 0x01;
   grid[_OFF(w, x-1, y-1)] = (cw >> 6) & 0x01;
   grid[_OFF(w, 0, y-1)] = (cw >> 7) & 0x01;
}

/******************************************************************************/
/** get special case 4
 * @param x width of region
 * @param y height of region
 */

static void _get4(dm_region_t *reg, int x, int y, uint8_t *cw)
{
   int w = reg->w;
   uint8_t *grid = reg->grid;


   *cw = grid[_OFF(w, x-1, 1)];
   *cw |= grid[_OFF(w, x-2, 1)] << 1;
   *cw |= grid[_OFF(w, x-3, 1)] << 2;
   *cw |= grid[_OFF(w, x-1, 0)] << 3;
   *cw |= grid[_OFF(w, x-2, 0)] << 4;
   *cw |= grid[_OFF(w, x-3, 0)] << 5;
   *cw |= grid[_OFF(w, x-1, y-1)] << 6;
   *cw |= grid[_OFF(w, 0, y-1)] << 7;
}

/******************************************************************************/
/** translate between a region and a buf
 * FIXME - add return to error if visit alloc fails
 */

static inline void grid_xlate(dm_region_t *grid, dm_buf_t *data)
{
   uint8_t mark = 0xff, *buf = &data->buf[data->idx];
   int h = grid->h, w = grid->w;
   int idx, x, y;
   dm_region_t *visit;


   /* alloc a temp grid to keep track of visited bits */
   if(!(visit = dm_region_new(w, h)))
      return;

   /* init starting spots */
   x = idx = 0;
   y = 4;

   for(; x < h || y < w;)
   {

      /* check special placement */
      if(!x)
      {
         if(y == h)
         {
            _put1(visit, w, h, &mark);
            _get1(grid, w, h, &(buf[idx++]));
         }
         else if(y == h - 2)
         {
            if(w % 4)
            {
               _put2(visit, w, h, &mark);
               _get2(grid, w, h, &(buf[idx++]));
            }
            else if(w % 8 == 4)
            {
               _put3(visit, w, h, &mark);
               _get3(grid, w, h, &(buf[idx++]));
            }
         }
      }
      else if(x == 2 && y == h + 4 && !(w % 8))
      {
         _put4(visit, w, h, &mark);
         _get4(grid, w, h, &(buf[idx++]));
      }

      /* move up and right */
      for(; x<w && y>=0; x+=2, y-=2)
         if(y < h && x >= 0 && !visit->grid[_OFF(w, x, y)])
            _get0(grid, x, y, &(buf[idx++]));

      x += 3;
      y++;

      /* move down and left */
      for(; x>=0 && y<h; x-=2, y+=2)
         if(y >= 0 && x < w && !visit->grid[_OFF(w, x, y)])
            _get0(grid, x, y, &(buf[idx++]));

      x++;
      y += 3;
   }

   data->idx += idx;
   free(visit);
}

static dm_region_t* dm_region_new(int w, int h)
{
   dm_region_t *grid;

   if(!(grid = calloc(1, sizeof(dm_region_t) + w * h)))
      return (dm_region_t *)PyErr_NoMemory();

   grid->w = w;
   grid->h = h;

   return grid;
}

/******************************************************************************/

static char* dm_decode(dm_buf_t *data)
{
   int digits, tens;
   char *output, *ptr;
   uint8_t *buf = data->buf;
   uint8_t *end = &buf[data->idx];
  

   if(!(ptr = output = malloc((data->idx <<2) + 1)))
      return (char *)PyErr_NoMemory();

   for(; buf<end; buf++)
   {
      /* normal ascii */
      if(*buf < 129)
         *ptr++ = *buf - 1;
      /* padding */
      else if(*buf == 129)
         break;
      /* pair of digits */
      else if(*buf < 230)
      {
         digits = *buf - 130;
         tens = digits / 10;
         *ptr++ = '0' + tens;
         *ptr++ = '0' + (digits - ((tens <<3) + (tens <<1)));
      }
      else
         return NULL;
   }

   /* null terminate */
   *ptr = 0;
   return output;
}

/******************************************************************************/
/** reads a data matrix and returns it's decoded string
 */

static char* dm_region_read(dm_region_t *reg, const dm_szdef_t *def, void *rs)
{
   int err;
   dm_buf_t *data;
   char *result;

   /* max size is number of bits / 8 */
   data = dm_buf_new((reg->w * reg->h) >> 3);
   if (! data)
      return NULL;

   grid_xlate(reg, data);
   
   data->idx = def->dcw + def->rscw;

   err = decode_rs_char(rs, data->buf, NULL, 0); /* in place error correction */
   if (err == -1) {
      PyErr_SetString(PyExc_IOError, "matrix error correction failed");
      return NULL;
   }

   data->idx = data->idx - def->rscw;
   
   result = dm_decode(data);
   free(data);
   if (! result)
      PyErr_SetString(PyExc_IOError, "matrix decoding failed");
   return result;
}

static void *set_width_height(int width, int height)
{
   const dm_szdef_t *def;

   // only re-init rs decoder if size has changed
   if (current_size && current_size->w == width && current_size->h == height)
      return rs_decoder;

   /* get definition for read region size */
   if (!(def = dm_szdef_wh(width, height))) {
      return NULL;
   }

   if (rs_decoder)
      free_rs_char(rs_decoder);

   rs_decoder = init_rs_char(RS_SYMSZ, 0x12d, 1, 1, def->rscw, RS_N - (def->dcw + def->rscw));
   current_size = def;

   return rs_decoder;
}

/******************************************************************************
 * Python binding
 ******************************************************************************/

static PyObject *decode(PyObject *self, PyObject *args);
static PyMethodDef DataMatrixMethods[] = {
   {"decode",  (PyCFunction)decode, METH_VARARGS, "Decodes a datamatrix encoded as a 2D list of booleans"},
   {NULL, NULL, 0, NULL}
};

static PyObject *decode(PyObject *self, PyObject *args)
{
   void *matrix;
   dm_region_t *reg;
   int x, y;
   int width, height;
   char *str;
   PyObject *row, *elem;

   if (! PyArg_ParseTuple(args, "O", &matrix))
      return NULL;

   height = PyList_Size(matrix);
   if (height < 0)
   {
      PyErr_SetString(PyExc_TypeError, "decode expects a 2D list");
      return NULL;
   }

   row = PyList_GetItem(matrix, 0);
   width = PyList_Size(row);
   if (width < 0)
   {
      PyErr_SetString(PyExc_TypeError, "decode expects a 2D list");
      return NULL;
   }

   if (! set_width_height(width, height))
   {
      PyErr_SetString(PyExc_TypeError, "decode: matrix width and height are invalid");
      return NULL;
   }

   reg = dm_region_new(width, height);
   if (! reg)
   {
      return NULL;
   }

   for (y = 0; y < height; y++)
   {
      row = PyList_GetItem(matrix, y);
      if (PyList_Size(row) != width) {
         PyErr_SetString(PyExc_TypeError, "decode: all list rows must have same length");
         free(reg);
         return NULL;
      }

      for (x = 0; x < width; x++)
      {
         elem = PyList_GetItem(row, x);
         reg->grid[y * width + x] = (elem == Py_True ? 0x01 : 0x00);
      }
   }

   str = dm_region_read(reg, current_size, rs_decoder);
   PyObject *ret = NULL;
   if (str) {
      ret = Py_BuildValue("s", str);
      free(str);
   }

   free(reg);

   return ret;
}

PyMODINIT_FUNC initdatamatrix()
{
    PyObject *m;

    m = Py_InitModule("datamatrix", DataMatrixMethods);
    if (m == NULL)
        return;
}
