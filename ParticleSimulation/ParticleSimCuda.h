#pragma once
#include "Particle.h"
#include "AbstractParticleSim.h"
#include <random>
#include "glm/geometric.hpp"

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

class ParticleSimCuda : public AbstractParticleSim
{
public:

	void MakeStep();
	void RandomizeParticles(glm::vec2 boundsSize, float startVelocity, float r);

	ParticleSimCuda(int paticlesCount);
	~ParticleSimCuda();

protected:
	Particle *prevFrameCuda, *curFrameCuda;


	cudaError_t err;
	void GetCudaFrames();
	void SetCudaFrames();
};

