#pragma once
#include "Particle.h"
#include <random>
#include "glm/geometric.hpp"

class AbstractParticleSim
{
#define ParticleSim_dt 0.01f

public:
	int paticlesCount;
	Particle *prevFrame, *curFrame;
	glm::vec2 boundsSize;
	float startVelocity;

	virtual void RandomizeParticles(glm::vec2 boundsSize, float startVelocity, float r);
	virtual ~AbstractParticleSim() {}

	virtual void MakeStep() {};
};

