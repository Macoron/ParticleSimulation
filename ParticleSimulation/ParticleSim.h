#pragma once
#include "AbstractParticleSim.h"
#include "Particle.h"
#include <random>
#include "glm/geometric.hpp"

class ParticleSim : public AbstractParticleSim
{
#define simulation_dt 0.01f;

public:
	bool useOpenMP;

	ParticleSim(int paticlesCount, bool useOpenMP);
	~ParticleSim();

	// сгенерировать наборы для отладки
	void GenerateTest01(glm::vec2 boundsSize);
	void GenerateTest02(glm::vec2 boundsSize);

	void MakeStep();

	static void ResolveCollision(Particle& p1, Particle& p2);
	void CheckBorders(Particle& p1);
};

