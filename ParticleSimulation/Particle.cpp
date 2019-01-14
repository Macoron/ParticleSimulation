#include "Particle.h"



Particle::Particle()
{
}

//проверяем пересечение частиц
bool Particle::IsOverlap(Particle p1, Particle p2)
{
	float dist = glm::distance(p1.pos, p2.pos);
	float rSum = p1.r + p2.r;

	return dist < rSum;
}