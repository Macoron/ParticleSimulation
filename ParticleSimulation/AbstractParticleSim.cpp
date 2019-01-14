#include "AbstractParticleSim.h"


void AbstractParticleSim::RandomizeParticles(glm::vec2 boundsSize, float startVelocity, float r)
{
	this->boundsSize = boundsSize;

	// инициализация рандома
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> xDist(-boundsSize.x + r, boundsSize.x - r);
	std::uniform_real_distribution<float> yDist(-boundsSize.y + r, boundsSize.y - r);

	// заполняем массив частиц
	for (int i = 0; i < paticlesCount; i++)
	{
		prevFrame[i].r = r;

		prevFrame[i].pos.x = xDist(mt);
		prevFrame[i].pos.y = yDist(mt);

		//генерируем начальный вектор скорости
		std::uniform_real_distribution<float> vDist(-1.f, 1.f);
		glm::vec2 V(vDist(mt), vDist(mt));
		// нормализуем его
		glm::vec2 nV = glm::normalize(V);

		// сохраняем итоговую скорость
		prevFrame[i].velocity = nV * startVelocity;
	}
}