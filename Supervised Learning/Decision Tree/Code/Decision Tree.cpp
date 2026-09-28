#include <stdio.h> 
#include <stdlib.h> 

//Dinh nghia kieu du lieu khach hang - Customer 
struct customer{
	int ID;       //Customer ID           (So thu tu)                       : [10000;14999] *Luu y: Khong dung de huan luyen    
	int A;        //Age                   (Tuoi)                            : [18;74]                                             0 
	int G;        //Gender                (Gioi tinh)                       : (Nam;Nu)->(1;0)                                     1 
	int T;        //Tenure                (So thang gan bo)                 : [1;72]                                              2 
	int CT;       //Contract type         (Loai hop dong)                   : (1 thang;1 nam;2 nam)->(0;1;2)                      3 
	int PB;       //Paperless billing     (Hoa don dien tu)                 : (Yes;No)->(1;0)                                     4 
	int IS;       //Internet service      (Dich vu Internet)                : (Fiber optic;DSL;No)->(2;1;0)                       5 
	int TS;       //Tech support          (Ho tro ky thuat)                 : (Yes;No;No internet service)->(1;0;2)               6 
	double MC;    //Monthly charges       (Cuoc phi hang thang)             : [0;+vc)                                             7 
	double TC;    //Total charges         (Tong cuoc phi da thanh toan)     : [0;+vc)                                             8 
	int C;        //Churn                 (Roi bo hay o lai)                : (Roi bo;O lai)->(1;0)                               
};                                                                                                                                

//Dinh nghia kieu du lieu node 
struct Node{                 
	int is_leaf;            //Kiem tra node, 1 neu la nut la, 0 neu la nut quyet dinh                                             
	int predict_class;      //Nhan du doan, 1 neu khach hang roi bo, 0 neu khach hang o lai                                       
	int feature;            //Dac trung dung de chia, gia tri chay trong doan [0;8] 
	double threshold;       //Nguong chia tuong ung voi dac trung 
	struct Node* left;      //Node tro toi nhanh ben trai  
	struct Node* right;     //Node tro toi nhanh ben phai 
}; 

//Rut ngan ten kieu du lieu de thuan tien cho viec code 
typedef struct customer customer; 
typedef struct Node Node; 

//Ham tim gia tri lon nhat cua cac phan tu trong 1 mang 
double max(double a[], int n){
	double max = a[0];
	for(int i = 1; i < n; i++){
		if(a[i] >= max){
			max = a[i]; 
		} 
	} 
	return max; 
} 

//Ham tinh binh phuong cua mot so thuc  
double mu2(double x){
	return x * x; 
}

//Phan hoach lomuto theo gia tri cac phan tu cua mang 
double lomuto(double a[], int left, int right){
	int i = left - 1, pivot = a[right];
	for(int j = left; j < right; j++){
		if(a[j] <= pivot){
		++i;
		double tmp = a[i];
		a[i] = a[j];
		a[j] = tmp; 
		} 
	} 
	++i;
	double tmp = a[i];
	a[i] = a[right];
	a[right] = tmp;
	return i; 
} 

//Quicksort - Sap xep mang so thuc theo thu tu tang dan 
void quickSort(double a[], int left, int right){
	if(left < right){
		int pos = lomuto(a, left, right);
		quickSort(a, left, pos - 1);
		quickSort(a, pos + 1, right); 
	} 
} 

//Phan hoach lomuto theo gia tri ID cua cac phan tu trong mang customer  
double partition(customer a[], int left, int right){
	int i = left - 1;
	int pivot = a[right].ID;
	for(int j = left; j < right; j++){
		if(a[j].ID <= pivot){
		++i;
		customer tmp = a[i];
		a[i] = a[j];
		a[j] = tmp; 
		} 
	} 
	++i;
	customer tmp = a[i];
	a[i] = a[right];
	a[right] = tmp;
	return i; 
} 

//Quicksort - Sap xep mang customer theo thu tu ID tang dan 
void Sort(customer a[], int left, int right){
	if(left < right){
		int pos = partition(a, left, right);
		Sort(a, left, pos - 1);
		Sort(a, pos + 1, right); 
	} 
} 

//Ham lay du lieu tu file text f va bo vao mang customer a 
void datafile_to_array(customer a[], int n, FILE *f){
	if(f == NULL){
		printf("Khong the mo file !\n"); 
	} 
	else{
		int i = 0, j = 0; 
		double p; 
		while(fscanf(f, "%lf", &p) != -1){
			if(i % 11 == 0){
				a[j].ID = (int)p; 
			} 
			else if(i % 11 == 1){
				a[j].A = (int)p; 
			}
			else if(i % 11 == 2){
				a[j].G = (int)p; 
			}
			else if(i % 11 == 3){
				a[j].T = (int)p; 
			}
			else if(i % 11 == 4){
				a[j].CT = (int)p; 
			}
			else if(i % 11 == 5){
				a[j].PB = (int)p; 
			}
			else if(i % 11 == 6){
				a[j].IS = (int)p; 
			}
			else if(i % 11 == 7){
				a[j].TS = (int)p; 
			}
			else if(i % 11 == 8){
				a[j].MC = p; 
			}
			else if(i % 11 == 9){
				a[j].TC = p; 
			}
			else if(i % 11 == 10){
				a[j].C = (int)p; 
				j++;
			} 
			i++;  
		} 
	} 
} 

//Ham in ra toan bo phan tu cua mang customer 
void print(customer a[], int n){
	for(int k = 0; k < n; k++){
		printf("%6d %3d %2d ", a[k].ID, a[k].A, a[k].G); 
		printf("%3d %2d %2d ", a[k].T, a[k].CT, a[k].PB);
		printf("%2d %2d ", a[k].IS, a[k].TS);
		printf("%7.2lf %8.2lf ", a[k].MC, a[k].TC);
		printf("%2d\n", a[k].C); 
	} 
} 

//Ham tao node la moi - Cap phat dong 
Node* create_node(){
    Node* node = (Node*)malloc(sizeof(Node));
    node->is_leaf = 0;
    node->predict_class = -1;
    node->feature = -1;
    node->threshold = 0.0;
    node->left = NULL;
    node->right = NULL;
    return node;
}

//Ham tinh Gini Impurity cua nhan goc 
double giniC(customer a[], int n){
	double cnt = 0; 
	for(int i = 0; i < n; i++){
		if(a[i].C == 1){
			cnt++; 
		} 
	} 
	double pos = cnt / n, neg = 1 - (cnt / n); 
	double gini = 1 - mu2(pos) - mu2(neg);
	return gini; 
} 

//Ham tinh do loi thong tin (theo thuoc do Gini) doi voi dac trung Age 
double infoGain_A(customer a[], int n, double threshold){
	double x = 0, y = 0;
	double x1 = 0, y1 = 0;
	double x0 = 0, y0 = 0;
	for(int i = 0; i < n; i++){
		if(a[i].A >= threshold && a[i].C == 0){
			y++;
			y0++; 
		}
		else if(a[i].A >= threshold && a[i].C == 1){
			y++;
			y1++; 
		}
		else if(a[i].A < threshold && a[i].C == 0){
			x++;
			x0++; 
		}
		else if(a[i].A < threshold && a[i].C == 1){
			x++;
			x1++; 
		} 
	} 
	double l = (x == 0) ? 0 : (1 - mu2(x0 / x) - mu2(x1 / x));
	double r = (y == 0) ? 0 : (1 - mu2(y0 / y) - mu2(y1 / y));
	double result = giniC(a, n) - (y / n) * r - (x / n) * l;
	return result; 
} 

//Ham tinh nguong tot nhat cua dac trung Age 
double threshold_A(customer a[], int n){
	double *age = (double*)malloc(n * sizeof(double));
	for(int i = 0; i < n; i++){
		age[i] = (double)(a[i].A); 
	} 
	quickSort(age, 0, n - 1);
	for(int i = 1; i < n; i++){
		if(age[i] == age[i - 1]){
			age[i] = -1; 
		} 
	} 
	quickSort(age, 0, n - 1);
	int cnt = 0; 
	for(int i = 0; i < n; i++){
		if(age[i] != -1){
			cnt++; 
		}
	} 
	double *ageA = (double*)malloc(cnt * sizeof(double)); 
	for(int i = 0; i < cnt; i++){
		ageA[i] = age[i + n - cnt]; 
	} 
	free(age); 
	double ageB[cnt - 1]; 
	double *ageC = (double*)malloc((cnt - 1) * sizeof(double)); 
	for(int i = 0; i < cnt - 1; i++){
		ageB[i] = (ageA[i] + ageA[i + 1]) / 2; 
		ageC[i] = infoGain_A(a, n, ageB[i]); 
	} 
	double max_infoGain = max(ageC, cnt - 1);
	int i;
	for(i = 0; i < cnt - 1; i++){
		if(ageC[i] == max_infoGain){
			break; 
		} 
	} 
	free(ageA);
	free(ageC); 
	return ageB[i]; 
} 

double infoGain_G(customer a[], int n){
	double female = 0, male = 0; 
	double female1 = 0, male1 = 0;
	double female0 = 0, male0 = 0;
	for(int i = 0; i < n; i++){
		if(a[i].G == 0 && a[i].C == 0){
			female++;
			female0++; 
		} 
		else if(a[i].G == 0 && a[i].C == 1){
			female++;
			female1++; 
		}
		else if(a[i].G == 1 && a[i].C == 1){
			male++;
			male1++; 
		}
		else if(a[i].G == 1 && a[i].C == 0){
			male++;
			male0++; 
		}
	} 
	double neg = (female == 0) ? 0 : (1 - mu2(female1 / female) - mu2(female0 / female));
	double pos = (male == 0) ? 0 : (1 - mu2(male1 / male) - mu2(male0 / male));
	double result = giniC(a, n) - (female / n) * neg - (male / n) * pos;
	return result; 
} 

double infoGain_T(customer a[], int n, double threshold){
	double x = 0, y = 0;
	double x1 = 0, y1 = 0;
	double x0 = 0, y0 = 0;
	for(int i = 0; i < n; i++){
		if(a[i].T >= threshold && a[i].C == 0){
			y++;
			y0++; 
		}
		else if(a[i].T >= threshold && a[i].C == 1){
			y++;
			y1++; 
		}
		else if(a[i].T < threshold && a[i].C == 0){
			x++;
			x0++; 
		}
		else if(a[i].T < threshold && a[i].C == 1){
			x++;
			x1++; 
		} 
	} 
	double l = (x == 0) ? 0 : (1 - mu2(x0 / x) - mu2(x1 / x));
	double r = (y == 0) ? 0 : (1 - mu2(y0 / y) - mu2(y1 / y));
	double result = giniC(a, n) - (y / n) * r - (x / n) * l;
	return result; 
} 

double threshold_T(customer a[], int n){
	double *tenure = (double*)malloc(n * sizeof(double));
	for(int i = 0; i < n; i++){
		tenure[i] = (double)a[i].T; 
	} 
	quickSort(tenure, 0, n - 1);
	for(int i = 1; i < n; i++){
		if(tenure[i] == tenure[i - 1]){
			tenure[i] = -1; 
		} 
	} 
	quickSort(tenure, 0, n - 1);
	int cnt = 0; 
	for(int i = 0; i < n; i++){
		if(tenure[i] != -1){
			cnt++; 
		}
	} 
	double *tenureA = (double*)malloc(cnt * sizeof(double)); 
	for(int i = 0; i < cnt; i++){
		tenureA[i] = tenure[i + n - cnt]; 
	} 
	free(tenure); 
	double tenureB[cnt - 1];
	double *tenureC = (double*)malloc((cnt - 1) * sizeof(double)); 
	for(int i = 0; i < cnt - 1; i++){
		tenureB[i] = (tenureA[i] + tenureA[i + 1]) / 2; 
		tenureC[i] = infoGain_T(a, n, tenureB[i]); 
	} 
	double max_infoGain = max(tenureC, cnt - 1);
	int i;
	for(i = 0; i < cnt - 1; i++){
		if(tenureC[i] == max_infoGain){
			break; 
		} 
	} 
	free(tenureA);
	free(tenureC); 
	return tenureB[i]; 
}

double infoGain_CT(customer a[], int n, int threshold){
	double th = 0, nth = 0; 
	double th0 = 0, nth0 = 0; 
	double th1 = 0, nth1 = 0; 
	for(int i = 0; i < n; i++){
		if(a[i].CT == threshold && a[i].C == 0){
			th++;
			th0++; 
		} 
		else if(a[i].CT == threshold && a[i].C == 1){
			th++;
			th1++; 
		}
		else if(a[i].CT != threshold && a[i].C == 0){
			nth++;
			nth0++; 
		}
		else if(a[i].CT != threshold && a[i].C == 1){
			nth++;
			nth1++; 
		}
	} 
	double th_ = (th == 0) ? 0 : (1 - mu2(th0 / th) - mu2(th1 / th));
	double nth_ = (nth == 0) ? 0 : (1 - mu2(nth0 / nth) - mu2(nth1 / nth)); 
	double result = giniC(a, n) - (th / n) * th_ - (nth / n) * nth_;
	return result; 
}

double threshold_CT(customer a[], int n){
	double cmp[3];
	cmp[0] = infoGain_CT(a, n, 0); 
	cmp[1] = infoGain_CT(a, n, 1);
	cmp[2] = infoGain_CT(a, n, 2);
	double max_info = max(cmp, 3);
	int i; 
	for(i = 0; i < 3; i++){
		if(cmp[i] == max_info){
			break; 
		} 
	} 
	return i; 
} 

double infoGain_PB(customer a[], int n){
	double yes = 0, no = 0; 
	double yes1 = 0, no1 = 0;
	double yes0 = 0, no0 = 0;
	for(int i = 0; i < n; i++){
		if(a[i].PB == 0 && a[i].C == 0){
			no++;
			no0++; 
		} 
		else if(a[i].PB == 0 && a[i].C == 1){
			no++;
			no1++; 
		}
		else if(a[i].PB == 1 && a[i].C == 1){
			yes++;
			yes1++; 
		}
		else if(a[i].PB == 1 && a[i].C == 0){
			yes++;
			yes0++; 
		}
	} 
	double neg = (no == 0) ? 0 : (1 - mu2(no1 / no) - mu2(no0 / no));
	double pos = (yes == 0) ? 0 : (1 - mu2(yes1 / yes) - mu2(yes0 / yes));
	double result = giniC(a, n) - (no / n) * neg - (yes / n) * pos;
	return result; 
}  

double infoGain_IS(customer a[], int n, int threshold){
	double th = 0, nth = 0; 
	double th0 = 0, nth0 = 0; 
	double th1 = 0, nth1 = 0; 
	for(int i = 0; i < n; i++){
		if(a[i].IS == threshold && a[i].C == 0){
			th++;
			th0++; 
		} 
		else if(a[i].IS == threshold && a[i].C == 1){
			th++;
			th1++; 
		}
		else if(a[i].IS != threshold && a[i].C == 0){
			nth++;
			nth0++; 
		}
		else if(a[i].IS != threshold && a[i].C == 1){
			nth++;
			nth1++; 
		}
	} 
	double th_ = (th == 0) ? 0 : (1 - mu2(th0 / th) - mu2(th1 / th));
	double nth_ = (th == 0) ? 0 : (1 - mu2(nth0 / nth) - mu2(nth1 / nth)); 
	double result = giniC(a, n) - (th / n) * th_ - (nth / n) * nth_;
	return result; 
} 

double threshold_IS(customer a[], int n){
	double cmp[3];
	cmp[0] = infoGain_IS(a, n, 0); 
	cmp[1] = infoGain_IS(a, n, 1);
	cmp[2] = infoGain_IS(a, n, 2);
	double max_info = max(cmp, 3);
	int i; 
	for(i = 0; i < 3; i++){
		if(cmp[i] == max_info){
			break; 
		} 
	} 
	return i; 
} 

double infoGain_TS(customer a[], int n, int threshold){
	double th = 0, nth = 0; 
	double th0 = 0, nth0 = 0; 
	double th1 = 0, nth1 = 0; 
	for(int i = 0; i < n; i++){
		if(a[i].TS == threshold && a[i].C == 0){
			th++;
			th0++; 
		} 
		else if(a[i].TS == threshold && a[i].C == 1){
			th++;
			th1++; 
		}
		else if(a[i].TS != threshold && a[i].C == 0){
			nth++;
			nth0++; 
		}
		else if(a[i].TS != threshold && a[i].C == 1){
			nth++;
			nth1++; 
		}
	} 
	double th_ = (th == 0) ? 0 : (1 - mu2(th0 / th) - mu2(th1 / th));
	double nth_ = (nth == 0) ? 0 : (1 - mu2(nth0 / nth) - mu2(nth1 / nth)); 
	double result = giniC(a, n) - (th / n) * th_ - (nth / n) * nth_;
	return result; 
} 

double threshold_TS(customer a[], int n){
	double cmp[3];
	cmp[0] = infoGain_TS(a, n, 0); 
	cmp[1] = infoGain_TS(a, n, 1);
	cmp[2] = infoGain_TS(a, n, 2);
	double max_info = max(cmp, 3);
	int i; 
	for(i = 0; i < 3; i++){
		if(cmp[i] == max_info){
			break; 
		} 
	} 
	return i; 
}

double infoGain_MC(customer a[], int n, double threshold){
	double x = 0, y = 0;
	double x1 = 0, y1 = 0;
	double x0 = 0, y0 = 0;
	for(int i = 0; i < n; i++){
		if(a[i].MC >= threshold && a[i].C == 0){
			y++;
			y0++; 
		}
		else if(a[i].MC >= threshold && a[i].C == 1){
			y++;
			y1++; 
		}
		else if(a[i].MC < threshold && a[i].C == 0){
			x++;
			x0++; 
		}
		else if(a[i].MC < threshold && a[i].C == 1){
			x++;
			x1++; 
		} 
	} 
	double l = (x == 0) ? 0 : (1 - mu2(x0 / x) - mu2(x1 / x));
	double r = (y == 0) ? 0 : (1 - mu2(y0 / y) - mu2(y1 / y));
	double result = giniC(a, n) - (y / n) * r - (x / n) * l;
	return result; 
} 

double threshold_MC(customer a[], int n){
	double *Mcharges = (double*)malloc(n * sizeof(double));
	for(int i = 0; i < n; i++){
		Mcharges[i] = a[i].MC; 
	} 
	quickSort(Mcharges, 0, n - 1);
	for(int i = 1; i < n; i++){
		if(Mcharges[i] == Mcharges[i - 1]){
			Mcharges[i] = -1; 
		} 
	} 
	quickSort(Mcharges, 0, n - 1);
	int cnt = 0; 
	for(int i = 0; i < n; i++){
		if(Mcharges[i] != -1){
			cnt++; 
		}
	} 
	double *MchargesA = (double*)malloc(cnt * sizeof(double)); 
	for(int i = 0; i < cnt; i++){
		MchargesA[i] = Mcharges[i + n - cnt]; 
	} 
	free(Mcharges); 
	double MchargesB[cnt - 1];
	double *MchargesC = (double*)malloc((cnt - 1) * sizeof(double)); 
	for(int i = 0; i < cnt - 1; i++){
		MchargesB[i] = (MchargesA[i] + MchargesA[i + 1]) / 2; 
		MchargesC[i] = infoGain_MC(a, n, MchargesB[i]); 
	} 
	double max_infoGain = max(MchargesC, cnt - 1);
	int i;
	for(i = 0; i < cnt - 1; i++){
		if(MchargesC[i] == max_infoGain){
			break; 
		} 
	} 
	free(MchargesA);
	free(MchargesC); 
	return MchargesB[i]; 
} 

double infoGain_TC(customer a[], int n, double threshold){
	double x = 0, y = 0;
	double x1 = 0, y1 = 0;
	double x0 = 0, y0 = 0;
	for(int i = 0; i < n; i++){
		if(a[i].TC >= threshold && a[i].C == 0){
			y++;
			y0++; 
		}
		else if(a[i].TC >= threshold && a[i].C == 1){
			y++;
			y1++; 
		}
		else if(a[i].TC < threshold && a[i].C == 0){
			x++;
			x0++; 
		}
		else if(a[i].TC < threshold && a[i].C == 1){
			x++;
			x1++; 
		} 
	} 
	double l = (x == 0) ? 0 : (1 - mu2(x0 / x) - mu2(x1 / x));
	double r = (y == 0) ? 0 : (1 - mu2(y0 / y) - mu2(y1 / y));
	double result = giniC(a, n) - (y / n) * r - (x / n) * l;
	return result; 
} 

double threshold_TC(customer a[], int n){
	double *Tcharges = (double*)malloc(n * sizeof(double));
	for(int i = 0; i < n; i++){
		Tcharges[i] = a[i].TC; 
	} 
	quickSort(Tcharges, 0, n - 1);
	for(int i = 1; i < n; i++){
		if(Tcharges[i] == Tcharges[i - 1]){
			Tcharges[i] = -1; 
		} 
	} 
	quickSort(Tcharges, 0, n - 1);
	int cnt = 0; 
	for(int i = 0; i < n; i++){
		if(Tcharges[i] != -1){
			cnt++; 
		}
	} 
	double *TchargesA = (double*)malloc(cnt * sizeof(double)); 
	for(int i = 0; i < cnt; i++){
		TchargesA[i] = Tcharges[i + n - cnt]; 
	} 
	free(Tcharges); 
	double TchargesB[cnt - 1];
	double *TchargesC = (double*)malloc((cnt - 1) * sizeof(double)); 
	for(int i = 0; i < cnt - 1; i++){
		TchargesB[i] = (TchargesA[i] + TchargesA[i + 1]) / 2; 
		TchargesC[i] = infoGain_TC(a, n, TchargesB[i]); 
	} 
	double max_infoGain = max(TchargesC, cnt - 1);
	int i;
	for(i = 0; i < cnt - 1; i++){
		if(TchargesC[i] == max_infoGain){
			break; 
		} 
	} 
	free(TchargesA);
	free(TchargesC); 
	return TchargesB[i]; 
}

int check_pure(customer a[], int n){
	int cnt = 0; 
	for(int i = 0; i < n; i++){
		if(a[i].C == 1){
			cnt++; 
		} 
	} 
	if(cnt == 0 || cnt == n){
		return 1; 
	} 
	return 0; 
} 

Node *BuildTree(customer a[], int n, int feature_, int depth, int max_depth, int min_split){
	Node *node = create_node(); 
	if(check_pure(a, n)){
		node->is_leaf = 1;
		node->predict_class = a[0].C;
		node->feature = -1;
		node->threshold = -1;
		node->left = NULL;
		node->right = NULL; 
		return node; 
	} 
	if(depth >= max_depth || n <= min_split){
		node->is_leaf = 1; 
		int cnt = 0;  
	    for(int i = 0; i < n; i++){
	    	if(a[i].C == 1){
	    		cnt++; 
			} 
		} 
		if(cnt >= n - cnt){
			node->predict_class = 1; 
		} 
		else{
			node->predict_class = 0; 
		} 
		node->feature = -1;
		node->threshold = -1;
		node->left = NULL;
		node->right = NULL;
		return node; 
	} 
	double feature[9]; 
	feature[0] = infoGain_A(a, n, threshold_A(a, n));
	feature[1] = infoGain_G(a, n);
	feature[2] = infoGain_T(a, n, threshold_T(a, n));
	feature[3] = infoGain_CT(a, n, threshold_CT(a, n));
	feature[4] = infoGain_PB(a, n);
	feature[5] = infoGain_IS(a, n, threshold_IS(a, n));
	feature[6] = infoGain_TS(a, n, threshold_TS(a, n));
	feature[7] = infoGain_MC(a, n, threshold_MC(a, n));
	feature[8] = infoGain_TC(a, n, threshold_TC(a, n));
	double maxi = max(feature, 9);
	int k;
	for(k = 0; k < 9; k++){
		if(feature[k] == maxi){
			break; 
		} 
	} 
	node->is_leaf = 0;
	node->predict_class = -1;
	node->feature = k;
	switch(k){
		case 0:
			node->threshold = threshold_A(a, n);
			break;
		case 1:
			node->threshold = 0.5; 
			break;
		case 2:
			node->threshold = threshold_T(a, n); 
			break; 
		case 3:
			node->threshold = threshold_CT(a, n); 
			break;
		case 4:
			node->threshold = 0.5; 
			break;
		case 5:
			node->threshold = threshold_IS(a, n); 
			break;
		case 6:
			node->threshold = threshold_TS(a, n); 
			break;
		case 7:
			node->threshold = threshold_MC(a, n); 
			break;
		default:
			node->threshold = threshold_TC(a, n); 
	} 
	customer ko;
	ko.ID = 0;
	ko.A = 0;
	ko.G = 0;
	ko.T = 0;
	ko.CT = 0;
	ko.PB = 0;
	ko.IS = 0;
	ko.TS = 0;
	ko.MC = 0;
	ko.TC = 0;
	ko.C = 0; 
	if(k == 0){
		int cnt = 0; 
		for(int i = 0; i < n; i++){
			if(a[i].A >= node->threshold){
				cnt++; 
			} 
		}  
		if(cnt == 0 || cnt == n){
			node->is_leaf = 1;
			node->predict_class = a[0].C;
			node->feature = -1;
			node->threshold = -1;
			node->left = NULL;
			node->right = NULL; 
			return node; 
		} 
		customer *tmp_r = (customer*)malloc(n * sizeof(customer));
		customer *tmp_l = (customer*)malloc(n * sizeof(customer)); 
		for(int i = 0; i < n; i++){
			tmp_r[i] = a[i]; 
			tmp_l[i] = a[i]; 
		} 
		customer right[cnt];
		customer left[n - cnt];
		for(int i = 0; i < n; i++){
			if(tmp_r[i].A < node->threshold){
				tmp_r[i] = ko; 
			} 
		} 
		for(int i = 0; i < n; i++){
			if(tmp_l[i].A >= node->threshold){
				tmp_l[i] = ko; 
			} 
		} 
		Sort(tmp_r, 0, n - 1); 
		Sort(tmp_l, 0, n - 1); 
		for(int i = 0; i < cnt; i++){
			right[i] = tmp_r[i + n - cnt];  
		} 
		for(int i = 0; i < n - cnt; i++){
			left[i] = tmp_l[i + cnt]; 
		}
		free(tmp_r);
		free(tmp_l); 
		node->right = BuildTree(right, cnt, 0, depth + 1, max_depth, min_split); 
		node->left = BuildTree(left, n - cnt, 0, depth + 1, max_depth, min_split); 
		return node; 
	}
	else if(k == 1){
		int cnt = 0;
		for(int i = 0; i < n; i++){
			if(a[i].G == 1){
				cnt++; 
			} 
		} 
		if(cnt == 0 || cnt == n){
			node->is_leaf = 1;
			node->predict_class = a[0].C;
			node->feature = -1;
			node->threshold = -1;
			node->left = NULL;
			node->right = NULL; 
			return node; 
		}  
		customer *tmp_r = (customer*)malloc(n * sizeof(customer));
		customer *tmp_l = (customer*)malloc(n * sizeof(customer)); 
		for(int i = 0; i < n; i++){
			tmp_r[i] = a[i]; 
			tmp_l[i] = a[i]; 
		} 
		customer right[cnt];
		customer left[n - cnt];
		for(int i = 0; i < n; i++){
			if(tmp_r[i].G == 0){
				tmp_r[i] = ko; 
			} 
		} 
		for(int i = 0; i < n; i++){
			if(tmp_l[i].G == 1){
				tmp_l[i] = ko; 
			} 
		} 
		Sort(tmp_r, 0, n - 1); 
		Sort(tmp_l, 0, n - 1); 
		for(int i = 0; i < cnt; i++){
			right[i] = tmp_r[i + n - cnt];  
		} 
		for(int i = 0; i < n - cnt; i++){
			left[i] = tmp_l[i + cnt]; 
		}
		free(tmp_r);
		free(tmp_l); 
		node->right = BuildTree(right, cnt, 1, depth + 1, max_depth, min_split); 
		node->left = BuildTree(left, n - cnt, 1, depth + 1, max_depth, min_split); 
		return node;
	}
	else if(k == 2){
		int cnt = 0; 
		for(int i = 0; i < n; i++){
			if(a[i].T >= node->threshold){
				cnt++; 
			} 
		}
		if(cnt == 0 || cnt == n){
			node->is_leaf = 1;
			node->predict_class = a[0].C;
			node->feature = -1;
			node->threshold = -1;
			node->left = NULL;
			node->right = NULL; 
			return node; 
		}    
		customer *tmp_r = (customer*)malloc(n * sizeof(customer));
		customer *tmp_l = (customer*)malloc(n * sizeof(customer));  
		for(int i = 0; i < n; i++){
			tmp_r[i] = a[i]; 
			tmp_l[i] = a[i]; 
		} 
		customer right[cnt];
		customer left[n - cnt];
		for(int i = 0; i < n; i++){
			if(tmp_r[i].T < node->threshold){
				tmp_r[i] = ko; 
			} 
		} 
		for(int i = 0; i < n; i++){
			if(tmp_l[i].T >= node->threshold){
				tmp_l[i] = ko; 
			} 
		} 
		Sort(tmp_r, 0, n - 1); 
		Sort(tmp_l, 0, n - 1); 
		for(int i = 0; i < cnt; i++){
			right[i] = tmp_r[i + n - cnt];  
		} 
		for(int i = 0; i < n - cnt; i++){
			left[i] = tmp_l[i + cnt]; 
		}
		free(tmp_r);
		free(tmp_l); 
		node->right = BuildTree(right, cnt, 2, depth + 1, max_depth, min_split); 
		node->left = BuildTree(left, n - cnt, 2, depth + 1, max_depth, min_split); 
		return node; 
	}
	else if(k == 3){
		int cnt = 0;
		for(int i = 0; i < n; i++){
			if(a[i].CT == node->threshold){
				cnt++; 
			} 
		} 
		if(cnt == 0 || cnt == n){
			node->is_leaf = 1;
			node->predict_class = a[0].C;
			node->feature = -1;
			node->threshold = -1;
			node->left = NULL;
			node->right = NULL; 
			return node; 
		} 
		customer *tmp_r = (customer*)malloc(n * sizeof(customer));
		customer *tmp_l = (customer*)malloc(n * sizeof(customer)); 
		for(int i = 0; i < n; i++){
			tmp_r[i] = a[i]; 
			tmp_l[i] = a[i]; 
		} 
		customer right[cnt];
		customer left[n - cnt];
		for(int i = 0; i < n; i++){
			if(tmp_r[i].CT != node->threshold){
				tmp_r[i] = ko; 
			} 
		} 
		for(int i = 0; i < n; i++){
			if(tmp_l[i].CT == node->threshold){
				tmp_l[i] = ko; 
			} 
		} 
		Sort(tmp_r, 0, n - 1); 
		Sort(tmp_l, 0, n - 1); 
		for(int i = 0; i < cnt; i++){
			right[i] = tmp_r[i + n - cnt];  
		} 
		for(int i = 0; i < n - cnt; i++){
			left[i] = tmp_l[i + cnt]; 
		}
		free(tmp_r);
		free(tmp_l); 
		node->right = BuildTree(right, cnt, 3, depth + 1, max_depth, min_split); 
		node->left = BuildTree(left, n - cnt, 3, depth + 1, max_depth, min_split); 
		return node;
	}
	else if(k == 4){
		int cnt = 0;
		for(int i = 0; i < n; i++){
			if(a[i].PB == 1){
				cnt++; 
			} 
		}
		if(cnt == 0 || cnt == n){
			node->is_leaf = 1;
			node->predict_class = a[0].C;
			node->feature = -1;
			node->threshold = -1;
			node->left = NULL;
			node->right = NULL; 
			return node; 
		}   
		customer *tmp_r = (customer*)malloc(n * sizeof(customer));
		customer *tmp_l = (customer*)malloc(n * sizeof(customer)); 
		for(int i = 0; i < n; i++){
			tmp_r[i] = a[i]; 
			tmp_l[i] = a[i]; 
		} 
		customer right[cnt];
		customer left[n - cnt];
		for(int i = 0; i < n; i++){
			if(tmp_r[i].PB == 0){
				tmp_r[i] = ko; 
			} 
		} 
		for(int i = 0; i < n; i++){
			if(tmp_l[i].PB == 1){
				tmp_l[i] = ko; 
			} 
		} 
		Sort(tmp_r, 0, n - 1); 
		Sort(tmp_l, 0, n - 1); 
		for(int i = 0; i < cnt; i++){
			right[i] = tmp_r[i + n - cnt];  
		} 
		for(int i = 0; i < n - cnt; i++){
			left[i] = tmp_l[i + cnt]; 
		}
		free(tmp_r);
		free(tmp_l); 
		node->right = BuildTree(right, cnt, 4, depth + 1, max_depth, min_split); 
		node->left = BuildTree(left, n - cnt, 4, depth + 1, max_depth, min_split); 
		return node;
	}
	else if(k == 5){
		int cnt = 0;
		for(int i = 0; i < n; i++){
			if(a[i].IS == node->threshold){
				cnt++; 
			} 
		} 
		if(cnt == 0 || cnt == n){
			node->is_leaf = 1;
			node->predict_class = a[0].C;
			node->feature = -1;
			node->threshold = -1;
			node->left = NULL;
			node->right = NULL; 
			return node; 
		}  
		customer *tmp_r = (customer*)malloc(n * sizeof(customer));
		customer *tmp_l = (customer*)malloc(n * sizeof(customer)); 
		for(int i = 0; i < n; i++){
			tmp_r[i] = a[i]; 
			tmp_l[i] = a[i]; 
		} 
		customer right[cnt];
		customer left[n - cnt];
		for(int i = 0; i < n; i++){
			if(tmp_r[i].IS != node->threshold){
				tmp_r[i] = ko; 
			} 
		} 
		for(int i = 0; i < n; i++){
			if(tmp_l[i].IS == node->threshold){
				tmp_l[i] = ko; 
			} 
		} 
		Sort(tmp_r, 0, n - 1); 
		Sort(tmp_l, 0, n - 1); 
		for(int i = 0; i < cnt; i++){
			right[i] = tmp_r[i + n - cnt];  
		} 
		for(int i = 0; i < n - cnt; i++){
			left[i] = tmp_l[i + cnt]; 
		}
		free(tmp_r);
		free(tmp_l); 
		node->right = BuildTree(right, cnt, 5, depth + 1, max_depth, min_split); 
		node->left = BuildTree(left, n - cnt, 5, depth + 1, max_depth, min_split); 
		return node;
	}
	else if(k == 6){
		int cnt = 0;
		for(int i = 0; i < n; i++){
			if(a[i].TS == node->threshold){
				cnt++; 
			} 
		} 
		if(cnt == 0 || cnt == n){
			node->is_leaf = 1;
			node->predict_class = a[0].C;
			node->feature = -1;
			node->threshold = -1;
			node->left = NULL;
			node->right = NULL; 
			return node; 
		}  
		customer *tmp_r = (customer*)malloc(n * sizeof(customer));
		customer *tmp_l = (customer*)malloc(n * sizeof(customer));  
		for(int i = 0; i < n; i++){
			tmp_r[i] = a[i]; 
			tmp_l[i] = a[i]; 
		} 
		customer right[cnt];
		customer left[n - cnt];
		for(int i = 0; i < n; i++){
			if(tmp_r[i].TS != node->threshold){
				tmp_r[i] = ko; 
			} 
		} 
		for(int i = 0; i < n; i++){
			if(tmp_l[i].TS == node->threshold){
				tmp_l[i] = ko; 
			} 
		} 
		Sort(tmp_r, 0, n - 1); 
		Sort(tmp_l, 0, n - 1); 
		for(int i = 0; i < cnt; i++){
			right[i] = tmp_r[i + n - cnt];  
		} 
		for(int i = 0; i < n - cnt; i++){
			left[i] = tmp_l[i + cnt]; 
		}
		free(tmp_r);
		free(tmp_l); 
		node->right = BuildTree(right, cnt, 6, depth + 1, max_depth, min_split); 
		node->left = BuildTree(left, n - cnt, 6, depth + 1, max_depth, min_split); 
		return node;
	}
	else if(k == 7){
		int cnt = 0; 
		for(int i = 0; i < n; i++){
			if(a[i].MC >= node->threshold){
				cnt++; 
			} 
		}  
		if(cnt == 0 || cnt == n){
			node->is_leaf = 1;
			node->predict_class = a[0].C;
			node->feature = -1;
			node->threshold = -1;
			node->left = NULL;
			node->right = NULL; 
			return node; 
		}  
		customer *tmp_r = (customer*)malloc(n * sizeof(customer));
		customer *tmp_l = (customer*)malloc(n * sizeof(customer)); 
		for(int i = 0; i < n; i++){
			tmp_r[i] = a[i]; 
			tmp_l[i] = a[i]; 
		} 
		customer right[cnt];
		customer left[n - cnt];
		for(int i = 0; i < n; i++){
			if(tmp_r[i].MC < node->threshold){
				tmp_r[i] = ko; 
			} 
		} 
		for(int i = 0; i < n; i++){
			if(tmp_l[i].MC >= node->threshold){
				tmp_l[i] = ko; 
			} 
		} 
		Sort(tmp_r, 0, n - 1); 
		Sort(tmp_l, 0, n - 1); 
		for(int i = 0; i < cnt; i++){
			right[i] = tmp_r[i + n - cnt];  
		} 
		for(int i = 0; i < n - cnt; i++){
			left[i] = tmp_l[i + cnt]; 
		}
		free(tmp_r);
		free(tmp_l); 
		node->right = BuildTree(right, cnt, 7, depth + 1, max_depth, min_split); 
		node->left = BuildTree(left, n - cnt, 7, depth + 1, max_depth, min_split); 
		return node; 
	}
	else if(k == 8){
		int cnt = 0; 
		for(int i = 0; i < n; i++){
			if(a[i].TC >= node->threshold){
				cnt++; 
			} 
		}  
		if(cnt == 0 || cnt == n){
			node->is_leaf = 1;
			node->predict_class = a[0].C;
			node->feature = -1;
			node->threshold = -1;
			node->left = NULL;
			node->right = NULL; 
			return node; 
		}  
		customer *tmp_r = (customer*)malloc(n * sizeof(customer));
		customer *tmp_l = (customer*)malloc(n * sizeof(customer));  
		for(int i = 0; i < n; i++){
			tmp_r[i] = a[i]; 
			tmp_l[i] = a[i]; 
		} 
		customer right[cnt];
		customer left[n - cnt];
		for(int i = 0; i < n; i++){
			if(tmp_r[i].TC < node->threshold){
				tmp_r[i] = ko; 
			} 
		} 
		for(int i = 0; i < n; i++){
			if(tmp_l[i].TC >= node->threshold){
				tmp_l[i] = ko; 
			} 
		} 
		Sort(tmp_r, 0, n - 1); 
		Sort(tmp_l, 0, n - 1); 
		for(int i = 0; i < cnt; i++){
			right[i] = tmp_r[i + n - cnt];  
		} 
		for(int i = 0; i < n - cnt; i++){
			left[i] = tmp_l[i + cnt]; 
		}
		free(tmp_r);
		free(tmp_l); 
		node->right = BuildTree(right, cnt, 8, depth + 1, max_depth, min_split); 
		node->left = BuildTree(left, n - cnt, 8, depth + 1, max_depth, min_split); 
		return node; 
	}
}

int predict(Node *node, customer x){
	if(node->is_leaf == 1){
		return node->predict_class; 
	} 
	int f = node->feature; 
	switch(f){
		case 0:
			if(x.A < node->threshold){
				return predict(node->left, x);
			}  
			else{
				return predict(node->right, x); 
			} 
			break; 
		case 1: 
			if(x.G < node->threshold){
				return predict(node->left, x);
			}  
			else{
				return predict(node->right, x); 
			}
			break;	
		case 2: 
			if(x.T < node->threshold){
				return predict(node->left, x);
			}  
			else{
				return predict(node->right, x); 
			}
			break;
		case 3: 
			if(x.CT != node->threshold){
				return predict(node->left, x);
			}  
			else{
				return predict(node->right, x); 
			}
			break;
		case 4: 
			if(x.PB < node->threshold){
				return predict(node->left, x);
			}  
			else{
				return predict(node->right, x); 
			}
			break;
		case 5: 
			if(x.IS != node->threshold){
				return predict(node->left, x);
			}  
			else{
				return predict(node->right, x); 
			}
			break;
		case 6: 
			if(x.TS != node->threshold){
				return predict(node->left, x);
			}  
			else{
				return predict(node->right, x); 
			}
			break;
		case 7: 
			if(x.MC < node->threshold){
				return predict(node->left, x);
			}  
			else{
				return predict(node->right, x); 
			}
			break;
		default: 
			if(x.TC < node->threshold){
				return predict(node->left, x);
			}  
			else{
				return predict(node->right, x); 
			}
	} 
} 

double Precision(customer a[], int p[], int n){
	int TP = 0;
	int Nh_p = 0;
	for(int i = 0; i < n; i++){
		if(p[i] == 1 && a[i].C == 1){
			TP++;
			Nh_p++;  
		} 
		else if(p[i] == 1 && a[i].C == 0){
			Nh_p++;  
		} 
	} 
	double precision = 1.0 * TP / Nh_p;
	return precision; 
} 

double Recall(customer a[], int p[], int n){
	int TP = 0;
	int N_p = 0;
	for(int i = 0; i < n; i++){
		if(p[i] == 1 && a[i].C == 1){
			TP++; 
			N_p++; 
		}  
		else if(p[i] == 0 && a[i].C == 1){
			N_p++; 
		}
	} 
	double recall = 1.0 * TP / N_p;
	return recall; 
} 

double Accuracy(customer a[], int p[], int n){
	int TP = 0, TN = 0, FP = 0, FN = 0;   
	for(int i = 0; i < n; i++){
		if(p[i] == 1 && a[i].C == 1){
			TP++;
		} 
		else if(p[i] == 1 && a[i].C == 0){
			FP++; 
		} 
		else if(p[i] == 0 && a[i].C == 1){
			FN++; 
		}
		else if(p[i] == 0 && a[i].C == 0){
			TN++;  
		}
	} 
	double accuracy = (double)(TN + TP) / (TN + TP + FN + FP);
	return accuracy; 
} 

double F1_score(customer a[], int p[], int n){
	double P = Precision(a, p, n);
	double R = Recall(a, p, n);
	double f1_score = 2 * P * R / (P + R);
	return f1_score; 
} 

void Confusion_matrix(customer a[], int p[], int n){
	int TP = 0, TN = 0, FP = 0, FN = 0; 
	int Nh_p = 0, Nh_m = 0, N_p = 0, N_m = 0;  
	for(int i = 0; i < n; i++){
		if(p[i] == 1 && a[i].C == 1){
			TP++;
			Nh_p++; 
			N_p++; 
		} 
		else if(p[i] == 1 && a[i].C == 0){
			FP++; 
			Nh_p++; 
			N_m++; 
		} 
		else if(p[i] == 0 && a[i].C == 1){
			FN++; 
			Nh_m++; 
			N_p++; 
		}
		else if(p[i] == 0 && a[i].C == 0){
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
	printf("|                  |   %3d    |   %3d    |   %4d   |\n", N_p, N_m, n);
	printf("|__________________|__________|__________|__________|\n");
} 

int main(){
	customer dataset[9000];
	customer datatest[1000]; 
	FILE *f;
	f = fopen("customer_dataset.txt", "r");
	datafile_to_array(dataset, 9000, f); 
	fclose(f);
	FILE *p; 
	p = fopen("customer_datatest.txt", "r");
	datafile_to_array(datatest, 1000, p);
	fclose(p); 
	Node *ptr = BuildTree(dataset, 9000, -1, 0, 15, 20);  //max_depth <-> min_split 
	int pre[1000]; 
	for(int i = 0; i < 1000; i++){
		pre[i] = predict(ptr, datatest[i]); 
	} 
	int cnt = 0; 
	for(int i = 0; i < 1000; i++){
		if(pre[i] != datatest[i].C){
			cnt++; 
		} 
	}  
	printf("Mo hinh du doan sai: %d mau tren 1000 mau !\n", cnt);
	printf("Ty le du doan sai : %.4lf\n", cnt * 1.0 / 1000); 
	printf("Precision : %.4lf\n", Precision(datatest, pre, 1000)); 
	printf("Recall : %.4lf\n", Recall(datatest, pre, 1000));
	printf("F1-score : %.4lf\n", F1_score(datatest, pre, 1000)); 
	printf("Accuracy : %.4lf\n", Accuracy(datatest, pre, 1000)); 
	Confusion_matrix(datatest, pre, 1000);
	return 0;
} 
