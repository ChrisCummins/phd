/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 /*
    ParamListGL
    - class derived from ParamList to do simple OpenGL rendering of a parameter list
*/

#ifndef PARAMGL_H
    #define PARAMGL_H

    #if defined(__APPLE__) || defined(MACOSX)
        #include <GLUT/glut.h>
    #else
        #include <GL/freeglut.h>
    #endif

    #include <param.h>

    void beginWinCoords();
    void endWinCoords();
    void glPrint(int x, int y, const char *s, void *font);
    void glPrintShadowed(int x, int y, const char *s, void *font, float *color);

    class ParamListGL : public ParamList {
        public:
          ParamListGL(char *name = "");

          void Render(int x, int y, bool shadow = false);
          bool Mouse(int x, int y, int button=GLUT_LEFT_BUTTON, int state=GLUT_DOWN);
          bool Motion(int x, int y);
          void Special(int key, int x, int y);

          void SetSelectedColor(float r, float g, float b) { text_col_selected[0] = r; text_col_selected[1] = g; text_col_selected[2] = b; }
          void SetUnSelectedColor(float r, float g, float b) { text_col_unselected[0] = r; text_col_unselected[1] = g; text_col_unselected[2] = b; }

          int bar_x;
          int bar_w;
          int bar_h;
          int text_x;
          int separation;
          int value_x;
          int font_h;
          int start_x, start_y;
          int bar_offset;

          float text_col_selected[3];
          float text_col_unselected[3];
          float text_col_shadow[3];
          float bar_col_outer[3];
          float bar_col_inner[3];

          void *font;
    };

#endif
