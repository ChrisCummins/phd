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


#include <GL/glew.h>
#include <GL/freeglut.h>
#include "oclRenderParticles.h"
#include <math.h>

#define GL_POINT_SPRITE_ARB               0x8861
#define GL_COORD_REPLACE_ARB              0x8862
#define GL_VERTEX_PROGRAM_POINT_SIZE_NV   0x8642

ParticleRenderer::ParticleRenderer()
: m_pos(0),
  m_numParticles(0),
  m_pointSize(1.0f),
  m_spriteSize(2.0f),
  m_vertexShader(0),
  m_pixelShader(0),
  m_program(0),
  m_texture(0),
  m_pbo(0),
  m_vboColor(0)
{
    _initGL();
}

ParticleRenderer::~ParticleRenderer()
{
    m_pos = 0;
}

void ParticleRenderer::setPositions(float *pos, int numParticles)
{
    m_pos = pos;
    m_numParticles = numParticles;
}

void ParticleRenderer::setColors(float *color, int numParticles)
{
    glBindBuffer(GL_ARRAY_BUFFER_ARB, m_vboColor);
    glBufferData(GL_ARRAY_BUFFER_ARB, numParticles * 4 * sizeof(float), color, GL_STATIC_DRAW_ARB);
    glBindBuffer( GL_ARRAY_BUFFER_ARB, 0);
}

void ParticleRenderer::setPBO(size_t pbo, int numParticles)
{
    m_pbo = pbo;
    m_numParticles = numParticles;
}

void ParticleRenderer::_drawPoints(bool color)
{
    if (!m_pbo)
    {
        glBegin(GL_POINTS);
        {
            int k = 0;
            for (int i = 0; i < m_numParticles; ++i)
            {
                glVertex3fv(&m_pos[k]);
                k += 4;
            }
        }
        glEnd();
    }
    else
    {
        glEnableClientState(GL_VERTEX_ARRAY);                
        
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, (unsigned int)m_pbo);
        glVertexPointer(4, GL_FLOAT, 0, 0);
        if (color)
        {
            glEnableClientState(GL_COLOR_ARRAY);
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, m_vboColor);
            //glActiveTexture(GL_TEXTURE1);
            //glTexCoordPointer(4, GL_FLOAT, 0, 0);
            glColorPointer(4, GL_FLOAT, 0, 0);
        }
        glDrawArrays(GL_POINTS, 0, m_numParticles);
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0);
        glDisableClientState(GL_VERTEX_ARRAY); 
        glDisableClientState(GL_COLOR_ARRAY); 
    }
}

void ParticleRenderer::display(DisplayMode mode /* = PARTICLE_POINTS */)
{
    switch (mode)
    {
        case PARTICLE_POINTS:
            glColor3f(1, 1, 1);
            glPointSize(m_pointSize);
            _drawPoints();
            break;
        case PARTICLE_SPRITES:
        default:
            {
                // setup point sprites
                glEnable(GL_POINT_SPRITE_ARB);
                glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
                glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
                glPointSize(m_spriteSize);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                glEnable(GL_BLEND);
                glDepthMask(GL_FALSE);

                glUseProgram(m_program);
                GLuint texLoc = glGetUniformLocation(m_program, "splatTexture");
                glUniform1i(texLoc, 0);

                glActiveTextureARB(GL_TEXTURE0_ARB);
                glBindTexture(GL_TEXTURE_2D, m_texture);

                glutReportErrors();

                glColor3f(1, 1, 1);
                
                _drawPoints();

                glUseProgram(0);

                glutReportErrors();

                glDisable(GL_POINT_SPRITE_ARB);
                glDisable(GL_BLEND);
                glDepthMask(GL_TRUE);
            }
            break;
        case PARTICLE_SPRITES_COLOR:
            {
                // setup point sprites
                glEnable(GL_POINT_SPRITE_ARB);
                glTexEnvi(GL_POINT_SPRITE_ARB, GL_COORD_REPLACE_ARB, GL_TRUE);
                glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
                glPointSize(m_spriteSize);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                glEnable(GL_BLEND);
                glDepthMask(GL_FALSE);

                glUseProgram(m_program);
                GLuint texLoc = glGetUniformLocation(m_program, "splatTexture");
                glUniform1i(texLoc, 0);

                glActiveTextureARB(GL_TEXTURE0_ARB);
                glBindTexture(GL_TEXTURE_2D, m_texture);

                glutReportErrors();

                glColor3f(1, 1, 1);
                
                _drawPoints(true);

                glUseProgram(0);

                glutReportErrors();

                glDisable(GL_POINT_SPRITE_ARB);
                glDisable(GL_BLEND);
                glDepthMask(GL_TRUE);
            }
            break;
    }
}

const char vertexShader[] = 
{    
    "void main()                                                            \n"
    "{                                                                      \n"
    "    float pointSize = 500.0 * gl_Point.size;                           \n"
	"    vec4 vert = gl_Vertex;												\n"
	"    vert.w = 1.0;														\n"
    "    vec3 pos_eye = vec3 (gl_ModelViewMatrix * vert);                   \n"
    "    gl_PointSize = max(1.0, pointSize / (1.0 - pos_eye.z));            \n"
    "    gl_TexCoord[0] = gl_MultiTexCoord0;                                \n"
    //"    gl_TexCoord[1] = gl_MultiTexCoord1;                                \n"
    "    gl_Position = ftransform();                                        \n"
    "    gl_FrontColor = gl_Color;                                          \n"
    "}                                                                      \n"
};

const char pixelShader[] =
{
    "uniform sampler2D splatTexture;                                        \n"
        
    "void main()                                                            \n"
    "{                                                                      \n"
    "    vec4 color = (0.6 + 0.4 * gl_Color) * texture2D(splatTexture, gl_TexCoord[0].st); \n"
    "    gl_FragColor =                                                     \n"
    "         color * mix(vec4(0.0, 0.2, 0.2, color.w), vec4(0.2, 0.7, 0.7, color.w), color.w);\n"
    "}                                                                      \n"
};

void ParticleRenderer::_initGL()
{
    m_vertexShader = glCreateShader(GL_VERTEX_SHADER);
    m_pixelShader = glCreateShader(GL_FRAGMENT_SHADER);

    const char* v = vertexShader;
    const char* p = pixelShader;
    glShaderSource(m_vertexShader, 1, &v, 0);
    glShaderSource(m_pixelShader, 1, &p, 0);
    
    glCompileShader(m_vertexShader);
    glCompileShader(m_pixelShader);

    m_program = glCreateProgram();

    glAttachShader(m_program, m_vertexShader);
    glAttachShader(m_program, m_pixelShader);

    glLinkProgram(m_program);

    _createTexture(32);

    glGenBuffers(1, (GLuint*)&m_vboColor);
    glBindBuffer( GL_ARRAY_BUFFER_ARB, m_vboColor);
    glBufferData( GL_ARRAY_BUFFER_ARB, m_numParticles * 4 * sizeof(float), 0, GL_STATIC_DRAW_ARB);
    glBindBuffer( GL_ARRAY_BUFFER_ARB, 0);

}

//------------------------------------------------------------------------------
// Function     	  : EvalHermite
// Description	    : 
//------------------------------------------------------------------------------
/**
* EvalHermite(float pA, float pB, float vA, float vB, float u)
* @brief Evaluates Hermite basis functions for the specified coefficients.
*/ 
inline float evalHermite(float pA, float pB, float vA, float vB, float u)
{
    float u2=(u*u), u3=u2*u;
    float B0 = 2*u3 - 3*u2 + 1;
    float B1 = -2*u3 + 3*u2;
    float B2 = u3 - 2*u2 + u;
    float B3 = u3 - u;
    return( B0*pA + B1*pB + B2*vA + B3*vB );
}

unsigned char* createGaussianMap(int N)
{
    float *M = new float[2*N*N];
    unsigned char *B = new unsigned char[4*N*N];
    float X,Y,Y2,Dist;
    float Incr = 2.0f/N;
    int i=0;  
    int j = 0;
    Y = -1.0f;
    //float mmax = 0;
    for (int y=0; y<N; y++, Y+=Incr)
    {
        Y2=Y*Y;
        X = -1.0f;
        for (int x=0; x<N; x++, X+=Incr, i+=2, j+=4)
        {
            Dist = (float)sqrtf(X*X+Y2);
            if (Dist>1) Dist=1;
            M[i+1] = M[i] = evalHermite(1.0f,0,0,0,Dist);
            B[j+3] = B[j+2] = B[j+1] = B[j] = (unsigned char)(M[i] * 255);
        }
    }
    delete [] M;
    return(B);
}    

void ParticleRenderer::_createTexture(int resolution)
{
    unsigned char* data = createGaussianMap(resolution);
    glGenTextures(1, (GLuint*)&m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, resolution, resolution, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
}
