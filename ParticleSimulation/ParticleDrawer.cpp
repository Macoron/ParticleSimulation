#include "ParticleDrawer.h"



ParticleDrawer::ParticleDrawer()
{
}


ParticleDrawer::~ParticleDrawer()
{
}

void ParticleDrawer::DrawSimulation(AbstractParticleSim& sim)
{
	/*Particle p;
	p.pos.x = 0;
	p.pos.y = 0;

	glRectf(p.pos.x - particleSize, p.pos.y - particleSize,
		p.pos.x + particleSize, p.pos.y + particleSize);*/

	for (int i = 0; i < sim.paticlesCount; i++)
	{
		Particle& p = sim.prevFrame[i];
		//printf("%f %f \n", p.pos.x, p.pos.y);
		glRectf(p.pos.x - particleSize, p.pos.y - particleSize,
			p.pos.x + particleSize, p.pos.y + particleSize);
	}
}
