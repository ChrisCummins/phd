/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "Log.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

extern int status;


void Sample::setMsg( const char *fmt, const char *msg )
{
    _isMsg = true;

    _fmt = new char[ strlen( fmt )+1 ];
    strcpy( _fmt, fmt );
    _msg = new char[ strlen( msg )+1 ];
    strcpy( _msg, msg );
}

void Sample::setTimer( const char *fmt, const char *msg, double timer, unsigned int nbytes, int loops )
{
    _isMsg = false;
    _timer = timer;

    if( loops != 0 ) _loops = loops;
    if( nbytes > 0 ) _nbytes = nbytes;

    if ( strlen( msg ) > 0 )
    {
       _fmt = new char[ strlen( fmt )+1 ];
       strcpy( _fmt, fmt );
    }

    if ( strlen( msg ) > 0 )
    {
        _msg = new char[ strlen( msg )+1 ];
        strcpy( _msg, msg );
    }
}

void Sample::printSample( void )
{
    if( _isMsg == true )
        printf( _fmt, _msg );
    else
    {
       double bwd = (((double) _nbytes * _loops )/ _timer ) / 1e9;

       printf( _fmt, _msg, _timer, bwd );
    }
}

TestLog::TestLog( int nSamples ) : _logIdx(0), 
                                   _logLoops(0), 
                                   _logLoopEntries(0),
                                   _logLoopTimers(0)
{
    _samples = new Sample[ nSamples ];
}

void TestLog::loopMarker()
{
    _logLoopEntries = 0;
    _logLoopTimers = 0;
    _logLoops++;
}

void TestLog::Msg( const char *format, const char *msg )
{
    _samples[ _logIdx++ ].setMsg( format, msg );
    _logLoopEntries++;
}

void TestLog::Error( const char *format, const char *msg )
{
    _samples[ _logIdx ].setMsg( format, msg );
    _samples[ _logIdx++ ].setErr();
    _logLoopEntries++;
}

void TestLog::Timer( const char *format, const char *msg, double timer, unsigned int nbytes, int loops )
{
    _samples[ _logIdx++ ].setTimer( format, msg, timer, nbytes, loops );
    _logLoopEntries++; 
    _logLoopTimers++;
}

void TestLog::printLog( void )
{
    int idx = 0;

    std::cout << "\nLOOP ITERATIONS\n";
    std::cout << "---------------\n";

    for(int l=0; l < _logLoops; l++)
    {
        std::cout << "\nLoop " << l << std::endl;

       for( int i=0; i < _logLoopEntries; i++ )
          _samples[ idx++ ].printSample();
    }
}

void TestLog::printSummary( int skip )
{
// return if a error or expected error has occured
    if(status)
    return;
    std::cout << "\nAVERAGES (over loops " << skip << " - " << _logLoops - 1 << ", use -l for complete log)\n";
    std::cout << "--------\n";

    for( int i = 0; i < _logLoopEntries; i++ )
    {
       if( _samples[ i ].isMsg() )
       {
           bool foundError = false;

           for( int nl = 0; nl < _logLoops; nl++ )
           {
               int current =  i + nl * _logLoopEntries;

               if( _samples[ current ].isErr() )
               {
                   _samples[ current ].printSample();
                   foundError = true;
                   break;
               }
           }

           if( !foundError )
               _samples[ i ].printSample();
       }
       else
       {
           double sum=0.;

           for( int nl = skip; nl < _logLoops; nl++ )
           {
               sum += _samples[ i + nl * _logLoopEntries ].getTimer();
           }

           _samples[ i ].setTimer( "", "", sum / ( _logLoops-skip ), 0, 0 );
           _samples[ i ].printSample();
       }
   }
}
