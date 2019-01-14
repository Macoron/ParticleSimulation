#include "ParticleSimCuda.h"

#define THREADS 32

ParticleSimCuda::ParticleSimCuda(int paticlesCount)
{
	this->paticlesCount = paticlesCount;

	prevFrame = new Particle[paticlesCount];
	curFrame = new Particle[paticlesCount];

	cudaMalloc(&prevFrameCuda, paticlesCount * sizeof(Particle));
	cudaMalloc(&curFrameCuda, paticlesCount * sizeof(Particle));
}

__device__ void CheckBorders(Particle & p1, glm::vec2 boundsSize)
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

__device__ bool IsOverlap(Particle p1, Particle p2)
{
	float dist = glm::distance(p1.pos, p2.pos);
	float rSum = p1.r + p2.r;

	return dist < rSum;
}

__device__ void ResolveCollision(Particle& p1, Particle& p2)
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

__global__ void cudaMoveAll(Particle *prevFrame, int paticlesCount, glm::vec2 boundsSize)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	Particle& curParticle = prevFrame[i];
	curParticle.pos += curParticle.velocity * ParticleSim_dt;
	CheckBorders(curParticle, boundsSize);
}

__global__ void resolveCollisionAll(Particle *prevFrame, Particle *curFrame)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	Particle curParticle = prevFrame[i];
	Particle& otherParticle = prevFrame[j];
	if (i != j)
	{
		if (IsOverlap(curParticle, otherParticle))
		{
			ResolveCollision(curParticle, otherParticle);
		}
	}
	//216/6
	curFrame[i] = curParticle;
}

void ParticleSimCuda::MakeStep()
{
	// передвиним все частицы
	cudaMoveAll << <paticlesCount/ THREADS, THREADS >> > (prevFrameCuda, paticlesCount, boundsSize);
	if (cudaPeekAtLastError() != cudaSuccess)
		printf("The error is %s", cudaGetErrorString(cudaGetLastError()));
	cudaDeviceSynchronize();

	// составим двухмерную матрицу всех возможных столкновений 
	dim3 blockSize(THREADS, THREADS);
	int grids = (paticlesCount + blockSize.x - 1) / blockSize.x;
	dim3 gridSize(grids, grids);

	// пересчитаем эти столкновения
	resolveCollisionAll << <gridSize, blockSize >> > (prevFrameCuda, curFrameCuda);
	cudaDeviceSynchronize();

	GetCudaFrames();

	// теперь переключимся на следующий кадр
	std::swap(prevFrameCuda, curFrameCuda);
}

void ParticleSimCuda::RandomizeParticles(glm::vec2 boundsSize, float startVelocity, float r)
{
	AbstractParticleSim::RandomizeParticles(boundsSize, startVelocity, r);
	SetCudaFrames();
}

ParticleSimCuda::~ParticleSimCuda()
{
	cudaFree(prevFrame);
	cudaFree(curFrame);

	delete[] prevFrame;
	delete[] curFrame;
}

void ParticleSimCuda::GetCudaFrames()
{
	cudaMemcpy(prevFrame, prevFrameCuda, paticlesCount * sizeof(Particle), cudaMemcpyDeviceToHost);
	cudaMemcpy(curFrame, curFrameCuda, paticlesCount * sizeof(Particle), cudaMemcpyDeviceToHost);
}

void ParticleSimCuda::SetCudaFrames()
{
	cudaMemcpy(prevFrameCuda, prevFrame, paticlesCount * sizeof(Particle), cudaMemcpyHostToDevice);
	cudaMemcpy(curFrameCuda, curFrame, paticlesCount * sizeof(Particle), cudaMemcpyHostToDevice);

}
