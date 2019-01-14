#define GLFW_INCLUDE_GLU

#include <GLFW/glfw3.h>
#include "AbstractParticleSim.h"
#include "ParticleSimCuda.h"
#include "ParticleDrawer.h"

#include <stdlib.h>
#include <stdio.h>
#include <chrono>

enum ParalelMode { SYNC, OPEN_MP, CUDA };
ParalelMode curMode = SYNC;

const int particleCounts = 512;
const glm::vec2 boundsSize(20, 20);
const float startVelocity = 10.f;

AbstractParticleSim* sim = nullptr;
ParticleDrawer drawer;

void updateSimulation()
{
	if (sim != nullptr)
		delete sim;

	switch (curMode)
	{
	case SYNC:
		sim = new ParticleSim(particleCounts, false);
		break;
	case OPEN_MP:
		sim = new ParticleSim(particleCounts, true);
		break;
	case CUDA:
		sim = new ParticleSimCuda(particleCounts);
		break;
	default:
		break;
	}

	sim->RandomizeParticles(boundsSize, startVelocity, 0.1f);
}


static void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error: %s\n", description);
}
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		switch (key)
		{
		case GLFW_KEY_R:
			sim->RandomizeParticles(boundsSize, startVelocity, 0.1f);
			break;
		case GLFW_KEY_1:
			curMode = ParalelMode::SYNC;
			updateSimulation();
			break;
		case GLFW_KEY_2:
			curMode = ParalelMode::OPEN_MP;
			updateSimulation();
			break;
		case GLFW_KEY_3:
			curMode = ParalelMode::CUDA;
			updateSimulation();
			break;
		default:
			break;
		}
	}
}

int main(void)
{
	updateSimulation();

	GLFWwindow* window;
	GLuint vertex_buffer, vertex_shader, fragment_shader, program;
	GLint mvp_location, vpos_location, vcol_location;
	glfwSetErrorCallback(error_callback);
	if (!glfwInit())
		exit(EXIT_FAILURE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwSetKeyCallback(window, key_callback);
	glfwMakeContextCurrent(window);
	


	std::chrono::high_resolution_clock clock;

	while (!glfwWindowShouldClose(window))
	{
		float ratio;
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		ratio = width / (float)height;

		glViewport(0, 0, width, height);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		if (width <= height) {
			gluOrtho2D(-boundsSize.x, boundsSize.x, 
				-boundsSize.y / ratio, boundsSize.y * ratio);
		}
		else {
			gluOrtho2D(-boundsSize.x * ratio, boundsSize.x * ratio,
				-boundsSize.y, boundsSize.y);
		}
		
		glClear(GL_COLOR_BUFFER_BIT);

		// делаем шаг в логике
		auto start = clock.now();
		sim->MakeStep();
		auto time = clock.now() - start;

		printf("Logic time %.3f ms \n", time.count() / 1e6);

		// отрисовываем
		start = clock.now();
		drawer.DrawSimulation(*sim);
		time = clock.now() - start;
		//printf("Rendering time %.3f ms \n", time.count() / 1e6);


		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}
