#include "AbstractParticleSim.h"


void AbstractParticleSim::RandomizeParticles(glm::vec2 boundsSize, float startVelocity, float r)
{
	this->boundsSize = boundsSize;

	// ������������� �������
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> xDist(-boundsSize.x + r, boundsSize.x - r);
	std::uniform_real_distribution<float> yDist(-boundsSize.y + r, boundsSize.y - r);

	// ��������� ������ ������
	for (int i = 0; i < paticlesCount; i++)
	{
		prevFrame[i].r = r;

		prevFrame[i].pos.x = xDist(mt);
		prevFrame[i].pos.y = yDist(mt);

		//���������� ��������� ������ ��������
		std::uniform_real_distribution<float> vDist(-1.f, 1.f);
		glm::vec2 V(vDist(mt), vDist(mt));
		// ����������� ���
		glm::vec2 nV = glm::normalize(V);

		// ��������� �������� ��������
		prevFrame[i].velocity = nV * startVelocity;
	}
}