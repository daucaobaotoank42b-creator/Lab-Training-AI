#include <stdio.h> 
#include <math.h> 
#include <stdlib.h> 
#include <string.h> 

typedef struct vector{
	double x;
	double y;
	int c; 
} vector; 

double min(double a[], int n){
	double min = a[0];
	for(int i = 1; i < n; i++){
		if(a[i] <= min){
			min = a[i]; 
		} 
	} 
	return min; 
} 

double mu2(double x){
	return x * x; 
}

void datafile_to_array(vector a[], int n, FILE *f){
	if(f == NULL){
		printf("Khong the mo file !\n"); 
	} 
	else{
		int i = 0, j = 0; 
		double p; 
		while(fscanf(f, "%lf", &p) != -1){
			if(i % 2 == 0){
				a[j].x = p; 
			} 
			else{
				a[j].y = p; 
				j++; 
			}
			i++;  
		} 
	} 
} 

double distance(vector a, vector b){                             
	double d = mu2(a.x - b.x) + mu2(a.y - b.y);
	return sqrt(d);
}

vector *kmeans_init_centroids(vector a[], int n, int k, int bias){       
	vector *centroids = (vector*)malloc(k * sizeof(vector));
	for(int i = 0; i < k; i++){
		centroids[i] = a[i + bias]; 
	} 
	return centroids; 
} 

void kmeans_assign_labels(vector a[], int n, int k, vector *c){         
	for(int i = 0; i < n; i++){
		double d[k];
		for(int j = 0; j < k; j++){
			d[j] = distance(a[i], c[j]); 
		} 
		int index;
		for(index = 0; index < k; index++){
			if(d[index] == min(d, k)){
				break; 
			} 
		} 
		a[i].c = index + 1; 
	} 
} 

void kmeans_update_centroids(vector a[], int n, int k, vector *centroids){    
    int p[k];
    vector v[k];
	vector ko;
	ko.x = 0, ko.y = 0, ko.c = 0;
	for(int i = 0; i < k; i++){
    	p[i] = 0; 
    	v[i] = ko; 
	}
	for(int i = 0; i < n; i++){
		for(int j = 0; j < k; j++){
			if(a[i].c == j + 1){
				p[j]++;
				v[j].x += a[i].x;
				v[j].y += a[i].y; 
			} 
		} 
	} 
	for(int i = 0; i < k; i++){
		if(p[i] > 0){
			centroids[i].x = v[i].x / p[i];
			centroids[i].y = v[i].y / p[i];
		} 
	} 
} 

int has_converged(vector a[], vector *b, int k){             
	int cnt = 0; 
	for(int i = 0; i < k; i++){
		if(distance(a[i], b[i]) == 0){
			cnt++; 
		}
	} 
	if(cnt == k){
		return 1; 
	} 
	return 0; 
} 

vector *kmeans(vector a[], int n,  int k, int bias){                          
	vector *centroid = kmeans_init_centroids(a, n, k, bias);
	int iter = 0;
	while(1){
		kmeans_assign_labels(a, n, k, centroid);
		vector tmp[k];
		for(int i = 0; i < k; i++){
			tmp[i] = centroid[i]; 
		} 
		kmeans_update_centroids(a, n, k, centroid); 
		if(has_converged(tmp, centroid, k)){
			break; 
		} 
		iter++;  
	}
	printf("%d\n", iter);
	return centroid; 
} 

void set_array(vector a[], int n, int k, vector b[][500]){
	vector ko;
	ko.x = 0, ko.y = 0, ko.c = 0;  
	for(int i = 0; i < k; i++){
		for(int j = 0; j < n; j++){
			if(a[j].c == i + 1){
				b[i][j] = a[j];
			} 
			else{
				b[i][j] = ko; 
			} 
		}
	}
} 

double loss_func(vector a[], int n, int k, vector b[][500], int bias, vector *centroid){ 
	double loss = 0;
	for(int i = 0; i < k; i++){
		for(int j = 0; j < n; j++){
			if(b[i][j].c != 0){
				loss += mu2(distance(centroid[i], b[i][j])); 
			} 
		} 
	} 
	return loss / n; 
} 

int main(){
	vector a[500];
	FILE *f;
	f = fopen("K-means_mix.txt", "r"); 
	datafile_to_array(a, 500, f); 
	fclose(f);
	int k;
	printf("Nhap so luong cluster K :"); 
	scanf("%d", &k); 
	int bias; 
	printf("Nhap bias :");
	scanf("%d", &bias);
	vector *centroid = kmeans(a, 500, k, bias);
	vector A[k][500];
	set_array(a, 500, k, A); 
	FILE **files = (FILE**)malloc(k * sizeof(FILE*)); 
	if(files == NULL){
        printf("Khong du bo nho!\n");
        return 1;
    }
	for(int i = 0; i < k; i++){
		char filename[] = "label"; 
		char stt[20];
		sprintf(stt, "%d", i + 1);
		strcat(filename, stt); 
		files[i] = fopen(filename, "w");
		for(int j = 0; j < 500; j++){
			if(A[i][j].c != 0){
				fprintf(files[i], "%20.15lf %20.15lf\n", A[i][j].x, A[i][j].y); 
			}
		} 
		fclose(files[i]);
	} 
	free(files);
	printf("%20.15lf", loss_func(a, 500, k, A, bias, centroid));
	return 0; 
} 
