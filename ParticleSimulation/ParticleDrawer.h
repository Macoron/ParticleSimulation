#pragma once
#include <GLFW/glfw3.h>
#include "ParticleSim.h"
#include "AbstractParticleSim.h"

class ParticleDrawer
{
public:
	ParticleDrawer();
	~ParticleDrawer();

	const float particleSize = 0.1f;
	void DrawSimulation(AbstractParticleSim& sim);
};

