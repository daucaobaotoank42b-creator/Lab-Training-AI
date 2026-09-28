#include <stdio.h>
#include <math.h> 
#include <stdlib.h> 

#define pi 3.141592654 

struct vector{
	double x;
	double y;
	int c; 
}; 

typedef struct vector vector;

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

double mean(double a[], int n){
	double sum = 0;
	for(int i = 0; i < n; i++){
		sum += a[i]; 
	} 
	return sum / n; 
} 

double standard_deviation(double a[], int n){
	double b = mean(a, n);
	double sum = 0; 
	for(int i = 0; i< n; i++){
		sum += pow(a[i] - b, 2); 
	} 
	return sqrt(sum / n); 
}

double func(double a[], int n, double x){
	double p = sqrt(2 * pi) * standard_deviation(a, n);
	double q = pow(x - mean(a, n), 2) / (2 * pow(standard_deviation(a, n), 2));
	return exp(-q) / p; 
} 

int predict_outlier(vector a[], int n, vector r, double epsilon){
	double fe1[n], fe2[n];
	for(int i = 0; i < n; i++){
		fe1[i] = a[i].x; 
		fe2[i] = a[i].y; 
	} 
	if(func(fe1, n, r.x) * func(fe2, n, r.y) < epsilon){
		return 1; 
	}
	return 0; 
} 

double Precision(vector a[], int p[], int n){
	int TP = 0;
	int Nh_p = 0;
	for(int i = 0; i < n; i++){
		if(p[i] == 1 && a[i].c == 1){
			TP++;
			Nh_p++;  
		} 
		else if(p[i] == 1 && a[i].c == 0){
			Nh_p++;  
		} 
	} 
	double precision = Nh_p > 0 ? (1.0 * TP / Nh_p) : 0;
	return precision; 
} 

double Recall(vector a[], int p[], int n){
	int TP = 0;
	int N_p = 0;
	for(int i = 0; i < n; i++){
		if(p[i] == 1 && a[i].c == 1){
			TP++; 
			N_p++; 
		}  
		else if(p[i] == 0 && a[i].c == 1){
			N_p++; 
		}
	} 
	double recall = N_p > 0 ? (1.0 * TP / N_p) : 0;
	return recall; 
} 

double Accuracy(vector a[], int p[], int n){
	int TP = 0, TN = 0, FP = 0, FN = 0;   
	for(int i = 0; i < n; i++){
		if(p[i] == 1 && a[i].c == 1){
			TP++;
		} 
		else if(p[i] == 1 && a[i].c == 0){
			FP++; 
		} 
		else if(p[i] == 0 && a[i].c == 1){
			FN++; 
		}
		else if(p[i] == 0 && a[i].c == 0){
			TN++;  
		}
	} 
	double accuracy = (double)(TN + TP) / (TN + TP + FN + FP);
	return accuracy; 
} 

double F1_score(vector a[], int p[], int n){
	double P = Precision(a, p, n);
	double R = Recall(a, p, n);
	double f1_score = (P + R) > 0 ? (2 * P * R / (P + R)) : 0;
	return f1_score; 
} 

void Confusion_matrix(vector a[], int p[], int n){
	int TP = 0, TN = 0, FP = 0, FN = 0; 
	int Nh_p = 0, Nh_m = 0, N_p = 0, N_m = 0;  
	for(int i = 0; i < n; i++){
		if(p[i] == 1 && a[i].c == 1){
			TP++;
			Nh_p++; 
			N_p++; 
		} 
		else if(p[i] == 1 && a[i].c == 0){
			FP++; 
			Nh_p++; 
			N_m++; 
		} 
		else if(p[i] == 0 && a[i].c == 1){
			FN++; 
			Nh_m++; 
			N_p++; 
		}
		else if(p[i] == 0 && a[i].c == 0){
			TN++; 
			Nh_m++;
			N_m++; 
		}
	} 
	printf("Confusion matrix :\n"); 
	printf(" ___________________________________________________\n");
	printf("|                  |        Truth        |          |\n");
	printf("|                  |_____________________|          |\n");
	printf("|                  |    1     |    0     |          |\n");
	printf("|------------------|----------|----------|----------|\n");
	printf("|            |  1  |   %3d    |   %3d    |   %3d    |\n", TP, FP, Nh_p);
	printf("|  Estimate  |-----|----------|----------|----------|\n");
	printf("|            |  0  |   %3d    |   %3d    |   %3d    |\n", FN, TN, Nh_m);
	printf("|------------------|----------|----------|----------|\n");
	printf("|                  |   %3d    |   %3d    |   %3d    |\n", N_p, N_m, n);
	printf("|__________________|__________|__________|__________|\n");
}

void save_file(vector pos[], int n, int m, vector neg[], double epsilon){
	int h[m];
	for(int i = 0; i < m; i++){
			h[i] = predict_outlier(pos, n, neg[i], epsilon);
	} 
	double precision[100]; 
	double recall[100]; 
	double f1_score[100]; 
	double accuracy[100];
	for(int k = 0; k < 100; k++){
		int pre[m];
		for(int i = 0; i < m; i++){
			pre[i] = predict_outlier(pos, n, neg[i], 0.0004 + k * 0.000396);
	    }
		precision[k] = Precision(neg, pre, m);
		recall[k] = Recall(neg, pre, m);
		f1_score[k] = F1_score(neg, pre, m);
		accuracy[k] = Accuracy(neg, pre, m); 
	}
	FILE *x;
	x = fopen("precision.txt", "w");
	for(int i = 0; i < 100; i++){
		fprintf(x, "%lf %lf\n", 0.0004 + i * 0.000396, precision[i]);
	} 
	fclose(x);
	FILE *y;
	y = fopen("recall.txt", "w");
	for(int i = 0; i < 100; i++){
		fprintf(y, " %lf %lf\n", 0.0004 + i * 0.000396, recall[i]);
	} 
	fclose(y);
	FILE *z;
	z = fopen("f1_score.txt", "w");
	for(int i = 0; i < 100; i++){
		fprintf(z, "%lf %lf\n", 0.0004 + i * 0.000396, f1_score[i]);
	} 
	fclose(z);
	FILE *t;
	t = fopen("accuracy.txt", "w");
	for(int i = 0; i < 100; i++){
		fprintf(t, "%lf %lf\n", 0.0004 + i * 0.000396, accuracy[i]);
	} 
	fclose(t);
} 

void print(vector a[], int n, int pre[]){
	printf("Precision : %lf\n", Precision(a, pre, n)); 
	printf("Recall : %lf\n", Recall(a, pre, n));
	printf("F1-score : %lf\n", F1_score(a, pre, n));
	printf("Accuracy : %lf\n", Accuracy(a, pre, n));
	Confusion_matrix(a, pre, n);
} 

int main(){
	vector datapos_train[500];
	vector dataneg_train[20]; 
	vector datapos_test[500];
	vector dataneg_test[20]; 
	double epsilon;
	scanf("%lf", &epsilon); 
	for(int i = 0; i < 500; i++){
		datapos_train[i].c = 0;
		datapos_test[i].c = 0; 
	} 
	for(int i = 0; i < 20; i++){
		dataneg_train[i].c = 1;
		dataneg_test[i].c = 1; 
	}
	FILE *f;
	f = fopen("inliers_500 train.txt", "r"); 
	datafile_to_array(datapos_train, 500, f);
	fclose(f);
	FILE *p;
	p = fopen("outliers_20 train.txt", "r"); 
	datafile_to_array(dataneg_train, 20, p);
	fclose(p);
	FILE *v;
	v = fopen("inliers_500 test.txt", "r"); 
	datafile_to_array(datapos_test, 500, v);
	fclose(v);
	FILE *g;
	g = fopen("outliers_20 test.txt", "r"); 
	datafile_to_array(dataneg_test, 20, g);
	fclose(g);
	save_file(datapos_test, 500, 20, dataneg_test, epsilon);
	int pre[20];
	for(int i = 0; i < 20; i++){
		pre[i] = predict_outlier(datapos_test, 500, dataneg_test[i], epsilon);
	}
	print(dataneg_test, 20, pre); 
	return 0; 
} 
