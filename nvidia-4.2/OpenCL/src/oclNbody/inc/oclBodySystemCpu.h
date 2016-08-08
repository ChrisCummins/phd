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

#ifndef __CL_BODYSYSTEMCPU_H__
    #define __CL_BODYSYSTEMCPU_H__

    #include "oclBodySystem.h"

    // CPU Body System
    class BodySystemCPU : public BodySystem
    {
        public:
            BodySystemCPU(int numBodies);
            virtual ~BodySystemCPU();

            virtual void update(float deltaTime);

            virtual void setSoftening(float softening) { m_softeningSquared = softening * softening; }
            virtual void setDamping(float damping)     { m_damping = damping; }

            virtual float* getArray(BodyArray array);
            virtual void   setArray(BodyArray array, const float* data);

            virtual size_t getCurrentReadBuffer() const { return m_currentRead; }

        protected: // methods
            BodySystemCPU() {} // default constructor

            virtual void _initialize(int numBodies);
            virtual void _finalize();

            void _computeNBodyGravitation();
            void _integrateNBodySystem(float deltaTime);
            
        protected: // data
            float* m_pos[2];
            float* m_vel[2];
            float* m_force;

            float m_softeningSquared;
            float m_damping;

            unsigned int m_currentRead;
            unsigned int m_currentWrite;
    };

#endif // __CL_BODYSYSTEMCPU_H__
