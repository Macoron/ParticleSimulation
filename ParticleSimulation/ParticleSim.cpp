#include "ParticleSim.h"

ParticleSim::ParticleSim(int paticlesCount, bool useOpenMP)
{
	this->paticlesCount = paticlesCount;
	this->useOpenMP = useOpenMP;
	prevFrame = new Particle[paticlesCount];
	curFrame = new Particle[paticlesCount];
}

ParticleSim::~ParticleSim()
{
	delete[] curFrame;
	delete[] prevFrame;
}

void ParticleSim::GenerateTest01(glm::vec2 boundsSize)
{
	this->paticlesCount = 2;

	prevFrame[0].pos.x = 2;
	prevFrame[0].pos.y = 2;
	prevFrame[1].pos.x = -2;
	prevFrame[1].pos.y = -2;

	prevFrame[0].velocity.x = -10;
	prevFrame[0].velocity.y = -10;
	prevFrame[1].velocity.x = 10;
	prevFrame[1].velocity.y = 10;
}

void ParticleSim::GenerateTest02(glm::vec2 boundsSize)
{
	this->paticlesCount = 2;

	prevFrame[0].pos.x = 2;
	prevFrame[0].pos.y = 0;
	prevFrame[1].pos.x = -2;
	prevFrame[1].pos.y = 0;

	prevFrame[0].velocity.x = 0;
	prevFrame[0].velocity.y = 0;
	prevFrame[1].velocity.x = 10;
	prevFrame[1].velocity.y = 0;
}

void ParticleSim::MakeStep()
{
	// передвиним все частицы
#pragma omp parallel for if (useOpenMP)
	for (int i = 0; i <paticlesCount; i++)
		prevFrame[i].pos += prevFrame[i].velocity * ParticleSim_dt;

#pragma omp parallel for if (useOpenMP)
	for (int i = 0; i < paticlesCount; i++)
	{
		Particle curParticle = prevFrame[i];

		// проверим что частицы не за границой мира
		CheckBorders(curParticle);

		// проверим на столкновения с другими частицами
		for (int j = 0; j < paticlesCount; j++)
		{
			if (i != j)
			{
				Particle otherParticle = prevFrame[j];
				if (Particle::IsOverlap(curParticle, otherParticle))
				{
					ResolveCollision(curParticle, otherParticle);
				}
			}
		}

		curFrame[i] = curParticle;
	}

	// теперь переключимся на следующий кадр
	std::swap(prevFrame, curFrame);
}

void ParticleSim::CheckBorders(Particle& p1)
{
	if (((p1.pos.x + p1.r) >= boundsSize.x))
	{
		p1.pos.x -= p1.r;
		p1.velocity.x *= -1.f;
	}
	else if (((p1.pos.x - p1.r) <= -boundsSize.x))
	{
		p1.pos.x += p1.r;
		p1.velocity.x *= -1.f;
	}

	if (((p1.pos.y + p1.r) >= boundsSize.y))
	{
		p1.pos.y -= p1.r;
		p1.velocity.y *= -1.f;
	}
	else if ((p1.pos.y - p1.r) <= -boundsSize.y)
	{
		p1.pos.y += p1.r;
		p1.velocity.y *= -1.f;
	}

}

void ParticleSim::ResolveCollision(Particle& p1, Particle& p2)
{
	float phi = atan2((p2.pos.y - p1.pos.y), (p2.pos.x - p1.pos.x));
	float speed1 = glm::length(p1.velocity);
	float speed2 = glm::length(p2.velocity);

	float direction_1 = atan2(p1.velocity.y, p1.velocity.x);
	float direction_2 = atan2(p2.velocity.y, p2.velocity.x);
	float new_xspeed_1 = speed1 * cos(direction_1 - phi);
	float new_yspeed_1 = speed1 * sin(direction_1 - phi);
	float new_xspeed_2 = speed2 * cos(direction_2 - phi);
	float new_yspeed_2 = speed2 * sin(direction_2 - phi);

	float final_xspeed_1 = ((p1.m - p2.m) * new_xspeed_1 + (p2.m + p2.m) * new_xspeed_2) / (p1.m + p2.m);
	float final_xspeed_2 = ((p1.m + p1.m) * new_xspeed_1 + (p2.m - p1.m) * new_xspeed_2) / (p1.m + p2.m);
	float final_yspeed_1 = new_yspeed_1;
	float final_yspeed_2 = new_yspeed_2;

	float cosAngle = cos(phi);
	float sinAngle = sin(phi);
	p1.velocity.x = cosAngle * final_xspeed_1 - sinAngle * final_yspeed_1;
	p1.velocity.y = sinAngle * final_xspeed_1 + cosAngle * final_yspeed_1;

	//p2.velocity.x = cosAngle * final_xspeed_2 - sinAngle * final_yspeed_2;
	//p2.velocity.y = sinAngle * final_xspeed_2 + cosAngle * final_yspeed_2;

	glm::vec2 pos1 = p1.pos;
	glm::vec2 pos2 = p2.pos;

	// get the mtd
	glm::vec2 posDiff = pos1 - pos2;
	float d = glm::length(posDiff);

	// push-pull them apart based off their mass
	//pos1 = pos1 + mtd * (im1 / (im1 + im2));
	auto mtd = glm::normalize(posDiff) * (p1.r + p2.r - d) * 1.f;
	p1.pos += mtd;

}