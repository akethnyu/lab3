#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#define H 1024
#define W 1024
#define C 3
#define Ir_index(c,j,i) (c * (H * W) + (j * W) + H)
#define O_index(c,j,i) (c * (H * W) + (j * W) + H)
#define size (H * W * C)
#define master 0

//int C1();
//int C2();

int main(int argc,char **argv)
{
        int world_size,world_rank;
        MPI_Init(NULL,NULL);
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Comm_size(MPI_COMM_WORLD,&world_size);
	MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
	double *Ir,*O,*Ir_recv;
	MPI_Request request,request_worker;
	MPI_Status status;
	double checksum = 0.0;

	int Or_size = sizeof(double) * C * H * W;
	int Ir_size = sizeof(double) * C * H * W; 

	if(world_rank == 0)
	{
		int tag = 0;

		O = (double *) malloc(Or_size);
		Ir_recv = (double *) malloc(Ir_size);

		for(int c = 0 ; c < C; c++)
                 for(int x = 0 ; x < W ; x++)
                  for(int y = 0 ; y < H ; y++)
                        {
                                Ir_recv[Ir_index(c,x,y)] = 0.0;
				O[O_index(c,x,y)] = 0.0;
                        }


		for(int i = 1; i < world_size; i++)
		{
			MPI_Irecv(Ir_recv,size,MPI_DOUBLE,i,tag,MPI_COMM_WORLD,&request);
			MPI_Wait(&request, &status);

			for(int c = 0 ; c < C; c++)
	                 for(int x = 0 ; x < W ; x++)
        	          for(int y = 0 ; y < H ; y++)
                          {
				if(c == 1 && x == 102 && y == 100) printf("Ir_recv is %d %f at C1 \n",i,Ir_recv[Ir_index(1,102,100)] );
                                O[O_index(c,x,y)] += Ir_recv[Ir_index(c,x,y)];
                                //O[O_index(c,x,y)] += i + c * (x + y);
				
                          }
		}

		for(int c = 0 ; c < C; c++)
                 for(int x = 0 ; x < W ; x++)
                  for(int y = 0 ; y < H ; y++)
                  {
			O[O_index(c,x,y)] /= world_size;
			checksum += O[O_index(c,x,y)];
                  }

		printf("Checksum in C1 is %f \n",checksum);
		printf("O at C1 is %f \n", O[O_index(1,102,100)] );

	}
	else
	{
		int tag = 0;
		Ir = (double *) malloc(Ir_size);

		for(int c = 0 ; c < C; c++)
		 for(int x = 0 ; x < W ; x++)
		  for(int y = 0 ; y < H ; y++)
			{
				Ir[Ir_index(c,x,y)] = world_rank + c * (x + y);
				if(c == 1 && x == 102 && y == 100) { printf("Ir is %d %f at C1 \n",world_rank,Ir[Ir_index(1,102,100)] ); }
			}

		MPI_Isend(Ir,size,MPI_DOUBLE,master,tag,MPI_COMM_WORLD,&request_worker);
		MPI_Wait(&request_worker, &status);
	}	

	MPI_Finalize();

}

/*
int C2()
{
	int world_size,world_rank;
        MPI_Comm_size(MPI_COMM_WORLD,&world_size);
        MPI_Comm_rank(MPI_COMM_WORLD,&world_rank);
        double *Ir,*O,*Ir_recv;
        MPI_Request request,request_worker;
        MPI_Status status;
	double checksum = 0.0;

	int Or_size = sizeof(double) * C * H * W;
        int Ir_size = sizeof(double) * C * H * W;
	
	Ir = (double *) malloc(Ir_size);
	O  = (double *)malloc(Or_size);

	if(world_rank == 0)
	{

		for(int c = 0 ; c < C; c++)
                 for(int x = 0 ; x < W ; x++)
                  for(int y = 0 ; y < H ; y++)
                        {
                                Ir[Ir_index(c,x,y)] = 0.0;
                        }
	}
	else
	{
		for(int c = 0 ; c < C; c++)
                 for(int x = 0 ; x < W ; x++)
                  for(int y = 0 ; y < H ; y++)
                        {
                                Ir[Ir_index(c,x,y)] = world_rank + c * (x + y);
				if(c == 1 && x == 102 && y == 100) printf("Ir is %d %f at C2 \n",world_rank,Ir[O_index(1,102,100)] );
                        }
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(Ir,O,size,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	for(int c = 0 ; c < C; c++)
                 for(int x = 0 ; x < W ; x++)
                  for(int y = 0 ; y < H ; y++)
                        {
                                O[O_index(c,x,y)] /= world_size;
				checksum += O[O_index(c,x,y)];
                        }
	printf("O at C2 is %f \n", O[O_index(1,102,100)] );
	printf("checksum in C2 is %f \n",checksum);
}
*/
