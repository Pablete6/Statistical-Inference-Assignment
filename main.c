#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N_IT 30000000

typedef struct {
	float vec[6];
} prob_t;

void init_rng (void)
{
	srand(time(NULL));
}

float urand (void)
{
	/* Return uniformly random number between
	   0 and 1 */
	return (float) rand() / RAND_MAX;
}

void make_prob (prob_t* p)
{
	/* Check no negative components */
	float norm = 0;
	for (int i = 0; i < 6; i++)
	{
		if (p->vec[i] < 0) p->vec[i] = 0;
		norm += p->vec[i];
	}

	/* Normalize */
	for (int i = 0; i < 6; i++)
	{
		p->vec[i] /= norm;
	}
}

float expected_value (const prob_t p)
{
	float ex_v = 0;
	for (int i = 0; i < 6; i++) ex_v += (i+1) * p.vec[i];

	return ex_v;
}

void init_prob (prob_t* p)
{
	do
	{
		float norm = 0;
		/* Random position in [0,1]^6 set */
		for (int i = 0; i < 6; i++)
		{
			p->vec[i] = urand();
		}
		
		/* Normalize (all components add up to one) */
		make_prob(p);
	} while (expected_value(*p) < 4);
}

void copy_prob (prob_t* dst_p,
		const prob_t src_p)
{
	for (int i = 0; i < 6; i++)
	{
		dst_p->vec[i] = src_p.vec[i];
	}
}

prob_t rnd_disp (const prob_t p,
		 const float dt)
{
	prob_t p_out;
	
	do
	{
		for (int i = 0; i < 6; i++)
		{
			p_out.vec[i] = p.vec[i] + 
				       dt * (2 * urand() - 1);
		}

		/* Check no negative values and renormalize */
		make_prob(&p_out);
	} while (expected_value(p_out) < 4);

	return p_out;
}

float obj_fun (const prob_t p)
{
	float out = 0;
	/* Kullback-Leibler divergence */
	for (int i = 0; i < 6; i++)
	{
		if (p.vec[i] != 0) out += p.vec[i] * logf(6 * p.vec[i]);
	}

	return out;
}

void montecarlo_step (prob_t* p,
		      const float temp,
		      size_t* acc)
{
	/* Make random displacement of type p */
	prob_t new_p = rnd_disp(*p, 0.01 * sqrtf(temp));

	/* Calculate difference in objective function */
	float df;

	df = obj_fun(new_p) - obj_fun(*p);

	/* Compute acceptance function and accept/reject
	   rew probability type */

	if (urand() < expf(- df / temp))
	{
		(*acc)++;
		copy_prob(p, new_p);
	}
}

void print_prob (const prob_t p)
{
	for (int i = 0; i < 6; i++)
	{
		printf("%.5f\t", p.vec[i]);
	}
	printf("\n");
}

int main()
{
	/* Initialize random type probabilities */
	prob_t p;
	init_rng(); // Initialize random number generator
	init_prob(&p);

	/* Find minimum value using annealing with 
	   metropolis montecarlo algorithm */
	size_t acc = 0;
	float temp = 1;
	for (size_t i = 0; i < N_IT; i++)
	{
		temp *= 0.999999;
		montecarlo_step(&p, temp, &acc);
	}

	/* Print result */
	
	printf("Optimal probability distribution is\n");
	print_prob(p);
	printf("\n\nAcceptance rate: %f (%lu)\n", (float) acc / N_IT, acc);
	printf("Expected value: %f, Kullback-Leibler divergence: "
	       "%f\n", expected_value(p), obj_fun(p));
	return 0;
}
