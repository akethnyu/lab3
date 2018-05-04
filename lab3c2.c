#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#define H 1024
#define W 1024
#define C 3
#define Ir_index(c,j,i) (c * (H * W) + (j * W) + i)
#define O_index(c,j,i) (c * (H * W) + (j * W) + i)
#define size (H * W * C)
#define master 0


int main(int argc,char **argv)
{
	int world_size,world_rank;
	MPI_Init(NULL,NULL);
        MPI_Comm_size(MPI_COMM_WORLD,&world_size);
        MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
        double *Ir,*O,*Ir_recv;
	double checksum = 0.0;

	int Or_size = sizeof(double) * C * H * W;
        int Ir_size = sizeof(double) * C * H * W;
	
	Ir = (double *) malloc(Ir_size);
	O  = (double *)malloc(Or_size);

	for(int c = 0 ; c < C; c++)
         for(int x = 0 ; x < W ; x++)
          for(int y = 0 ; y < H ; y++)
          {
          	Ir[Ir_index(c,x,y)] = world_rank + c * (x + y);
          }

	MPI_Allreduce(Ir,O,size,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	if(world_rank == 0)
	{
		for(int c = 0 ; c < C; c++)
                 for(int x = 0 ; x < W ; x++)
                  for(int y = 0 ; y < H ; y++)
                        {
                                O[O_index(c,x,y)] /= world_size;
				checksum += O[O_index(c,x,y)];
                        }

		printf("checksum in C2 is %f \n",checksum);
	}
	MPI_Finalize();

	free(Ir);
	free(O);

}
