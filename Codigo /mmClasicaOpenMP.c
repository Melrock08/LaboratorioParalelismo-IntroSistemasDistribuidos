/*#######################################################################################
 #* Fecha: 15 Agosto 2025
 #* Autor: Juan Santiago Saavedra Holguin
 #* Tema: 
 #* 	- Programa Multiplicación de Matrices algoritmo clásico
 #* 	- Paralelismo con OpenMP
######################################################################################*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>

// Declaracion de variables globales para el tiempo
struct timeval inicio, fin; 


// Esta funcion empieza a contar el tiempo de la muestra con gettimeofday() 
void InicioMuestra(){
	gettimeofday(&inicio, (void *)0);
}


// Esta funcion toma el tiempo final de la muestra y lo imprime en microsegundos

void FinMuestra(){
	gettimeofday(&fin, (void *)0);
	fin.tv_usec -= inicio.tv_usec;
	fin.tv_sec  -= inicio.tv_sec;
	double tiempo = (double) (fin.tv_sec*1000000 + fin.tv_usec); 
	printf("%9.0f \n", tiempo);
}


/* Esta funcion imprime la matriz cuadrada con el puntero a la matriz y el tamaño que se le pasan.
   Solo imprime si el tamaño es menor a 9 */
void impMatrix(double *matrix, int D){
	printf("\n");
	if(D < 9){
		for(int i=0; i<D*D; i++){
			if(i%D==0) printf("\n");
			printf("%.2f  ", matrix[i]);
		}
		printf("\n**-----------------------------**\n");
	}
}

// Esta funcion inicializa las matrices con valores aleatorios

void iniMatrix(double *m1, double *m2, int D){
	for(int i=0; i<D*D; i++, m1++, m2++){
		*m1 = (double)(rand()%100);	
		*m2 = (double)(rand()% 100);	
	}
}

// Esta funcion se encarga de multiplicar las dos matrices usando el paralelismo con OpenMP

void multiMatrix(double *mA, double *mB, double *mC, int D){
	double Suma, *pA, *pB;
	#pragma omp parallel
	{
	#pragma omp for
	for(int i=0; i<D; i++){
		for(int j=0; j<D; j++){
			pA = mA+i*D;	
			pB = mB+j; 	
			Suma = 0.0;
			for(int k=0; k<D; k++, pA++, pB+=D){
				Suma += *pA * *pB;
			}
			mC[i*D+j] = Suma;
		}
	}
	}
}

/* Main principal 
	- Este recibe dos argumentos cuando se ejecuta por consola o cmd : 
		1) el tamaño de la matriz  
		2) el numero de hilos  
*/

int main(int argc, char *argv[]){

	/* Verificacion de argumentos
		- Es obligatorio escribir los 2 argumentos que requiere el programa */
	if(argc < 3){
		printf("\n Use: $./clasicaOpenMP SIZE Hilos \n\n");
		exit(0);
	}


	int N = atoi(argv[1]);
	int TH = atoi(argv[2]);
	double *matrixA  = (double *)calloc(N*N, sizeof(double));
	double *matrixB  = (double *)calloc(N*N, sizeof(double));
	double *matrixC  = (double *)calloc(N*N, sizeof(double));
	srand(time(NULL));

	omp_set_num_threads(TH);

	iniMatrix(matrixA, matrixB, N);

	impMatrix(matrixA, N);
	impMatrix(matrixB, N);

	InicioMuestra();
	multiMatrix(matrixA, matrixB, matrixC, N);
	FinMuestra();

	impMatrix(matrixC, N);

	/*Liberación de Memoria*/
	free(matrixA);
	free(matrixB);
	free(matrixC);
	
	return 0;
}
