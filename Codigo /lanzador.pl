#!/usr/bin/perl
#**************************************************************
#         		Pontificia Universidad Javeriana
#     Autor: Juan Saavedra
#     Fecha: Agosto 2025
#     Materia: Sistemas Operativos
#     Tema: Taller de EvaluaciÃ³n de Rendimiento
#     Fichero: script automatizaciÃ³n ejecuciÃ³n por lotes 
#****************************************************************/

$Path = `pwd`;
chomp($Path);

# Nombre del ejecutable que se va a utilizar
$Nombre_Ejecutable = "mmClasicaOpenMP";

# Tamaños de las matrices para el experimento
# Nota: las pruebas van hasta el tamaño 8800 con 2 hilos por temas de tiempo
@Size_Matriz = ("800","1600","2400","3200","4000","4800","5600","6400","7200","8000","8800","9600");

# Numeros de hilos para el experimento
@Num_Hilos = (1,2,4,8,16,20);

# Numero de repeticiones por cada hilo
$Repeticiones = 30;


# Este bucle anidado recorre los tamaños de las matrices,
# despues recorre todos los hilos por cada tamaño,
# luego se ejecuta el programa segun el numero de repeticiones,
# finalmente se guardan los resultados los archivos con la siguiente estructura: 
# $Nombre_Ejecutable-".$size."-Hilos-".$hilo.".dat 
foreach $size (@Size_Matriz){
	foreach $hilo (@Num_Hilos) {
		$file = "$Path/$Nombre_Ejecutable-".$size."-Hilos-".$hilo.".dat";
		for ($i=0; $i<$Repeticiones; $i++) {
		system("$Path/$Nombre_Ejecutable $size $hilo  >> $file");
		printf("$Path/$Nombre_Ejecutable $size $hilo \n");
		}
		close($file);
	$p=$p+1;
	}
}
