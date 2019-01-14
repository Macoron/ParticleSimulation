#pragma once

#include <glm/vec2.hpp>
#include <glm/geometric.hpp>

class Particle
{
public:
	glm::vec2 pos;
	glm::vec2 velocity;
	float r = 0.1f;
	float m = 1.f;

	Particle();

	static bool IsOverlap(Particle p1, Particle p2);
};

