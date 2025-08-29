GCC = gcc
CFLAGS = -lm
FOPENMP = -fopenmp -O3
POSIX = -lpthread

modulo = inter.c

PROGRAMAS = mmClasicaOpenMP 

ALL: $(PROGRAMAS)


mmClasicaOpenMP:
	$(GCC) $(CLFAGS) $@.c -o $@ $(FOPENMP)

clean:
	$(RM) $(PROGRAMAS) 
