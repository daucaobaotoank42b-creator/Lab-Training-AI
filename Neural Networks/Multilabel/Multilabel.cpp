#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h> 

double **matrix(int m, int n){
    double **a = (double**)malloc(m * sizeof(double*));
    for(int i = 0; i < m; i++){
        a[i] = (double*)malloc(n * sizeof(double));
    }
    return a;
}

void free_matrix(double **a, int m){
    for(int i = 0; i < m; i++){
    	free(a[i]);
	} 
    free(a);
}

double **zero_matrix(int m, int n){
    double **Z = matrix(m, n);
    for(int i = 0; i < m; i++){
    	for(int j = 0; j < n; j++){
    		Z[i][j] = 0;
		}  
	} 
    return Z;
}

void equal(double **a, int m, int n, double **b){
    for(int i = 0; i < m; i++){
    	for(int j = 0; j < n; j++){
    		a[i][j] = b[i][j];
		}
	}      
}

double **matrix_multi(double **a, int m, int n, double **b, int q){
    double **c = zero_matrix(m, q);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < q; j++){
            for(int k = 0; k < n; k++){
                c[i][j] += a[i][k] * b[k][j];  
            }                
        }
    }
    return c;
}

double **scalar_multi(double **a, int m, int n, double x){
    double **X = matrix(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            X[i][j] = x * a[i][j];
        }           
    }
    return X;
}


double **transpose(double **a, int m, int n){
    double **aT = matrix(n, m);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < m; j++){
            aT[i][j] = a[j][i]; 
        }          
    }
    return aT;
}

double **add_bias(double **a, int m, int n, double **b){
    double **c = matrix(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            c[i][j] = a[i][j] + b[i][0];
        }
    }       
    return c;
}

double **matrix_add(double **a, int m, int n, double **b){
    double **c = matrix(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            c[i][j] = a[i][j] + b[i][j];
        }
    }      
    return c;
}

double **matrix_minus(double **a, int m, int n, double **b){
    double **c = matrix(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            c[i][j] = a[i][j] - b[i][j];
        }
    }     
    return c;
}

double min_arr(double *x, int n){
    double mi = x[0];
    for(int i = 1; i < n; i++){
        if(x[i] < mi){
            mi = x[i];
        } 
    } 
    return mi;
}

double max_arr(double *x, int n){
    double ma = x[0];
    for(int i = 1; i < n; i++){
       if(x[i] > ma){
           ma = x[i];
       }  
    } 
    return ma;
}

void scale_1d(double *x, int n, double *out_mi, double *out_ma){
    double mi = min_arr(x, n);
    double ma = max_arr(x, n);
    *out_mi = mi; *out_ma = ma;
    if(ma - mi == 0){
        return;
    }  
    for(int i = 0; i < n; i++){
        x[i] = (x[i] - mi) / (ma - mi);
    } 
}

void scale_2d(double **x, int m, int n, double *out_mi, double *out_ma){
    for(int i = 0; i < m; i++){
        double mi = min_arr(x[i], n);
        double ma = max_arr(x[i], n);
        out_mi[i] = mi; out_ma[i] = ma;
        if(ma - mi == 0){
            continue;
        } 
        for(int j = 0; j < n; j++){
        	x[i][j] = (x[i][j] - mi) / (ma - mi);
		} 
    }
}

void restore_1d(double *x, int n, double mi, double ma){
    for(int i = 0; i < n; i++){
        x[i] = x[i] * (ma - mi) + mi;
    } 
}

void apply_scale_1d(double *x, int n, double mi, double ma){
    if(ma - mi == 0){
        return;
    } 
    for(int i = 0; i < n; i++){
        x[i] = (x[i] - mi) / (ma - mi);
    } 
}

void binary_vector(double **a, int m, int n, double threshold){ 	
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			if(a[i][j] >= threshold){
				a[i][j] = 1; 
			} 
			else{
				a[i][j] = 0;
			} 
		} 
	}  
}

double LeakyReLU(double x){
    return (x > 0) ? x : 0.1 * x;
}

double gradLeakyReLU(double x){
    return (x > 0) ? 1.0 : 0.1;   
}

double **activation_matrix_LeakyReLU(double **a, int m, int n){
    double **b = matrix(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            b[i][j] = LeakyReLU(a[i][j]);
        }
    }     
    return b;
}

double Sigmoid(double x){
	return 1.0 / (1 + exp(-x)); 
} 

double gradSigmoid(double x){
	return Sigmoid(x) * (1 - Sigmoid(x)); 
} 

double **activation_matrix_Sigmoid(double **a, int m, int n){
    double **b = matrix(m, n);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            b[i][j] = Sigmoid(a[i][j]);
        }
    }    
    return b;
} 

struct Layer{
    int size_in;
    int size_out;
    double **w;   // (size_in  x size_out) 
    double **b;   // (size_out x 1)       
    double **z;   // (size_out x n_train)  
    double **a;   // (size_out x n_train)  
    double **e;   // (size_out x n_train)  
    double **dw;  // (size_in  x size_out) 
    double **db;  // (size_out x 1) 
    double **mw, **vw;   // (size_in  x size_out) 
    double **mb, **vb;   // (size_out x 1) 
};

typedef struct Layer Layer;

struct NeuralNetworks{
    int layer;        // so luong layer 
    int n_train;      // so mau du lieu train 
    int *adam_t;
    Layer *Layers;
};

typedef struct NeuralNetworks NeuralNetworks;

void random_matrix(double **x, int m, int n, double limit){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            x[i][j] = (2.0 * rand() / RAND_MAX - 1) * limit;
        }
    } 
}

NeuralNetworks create_network(int input_dim, int *hidden_sizes, int num_hidden, int output_dim, int n_train){
    NeuralNetworks net;
    net.layer = num_hidden + 1;  
    net.n_train = n_train;
    net.adam_t = (int*)malloc(sizeof(int));
    *net.adam_t = 0;
    net.Layers = (Layer*)malloc(net.layer * sizeof(Layer));
    int prev_size = input_dim;
    for(int i = 0; i < net.layer; i++){
        int this_size = (i < num_hidden) ? hidden_sizes[i] : output_dim;
        Layer *L = &net.Layers[i];
        L->size_in  = prev_size;
        L->size_out = this_size;
        L->w  = matrix(L->size_in, L->size_out);
        L->b  = matrix(L->size_out, 1);
        L->z  = matrix(L->size_out, n_train);
        L->a  = matrix(L->size_out, n_train);
        L->e  = matrix(L->size_out, n_train);
        L->dw = matrix(L->size_in, L->size_out);
        L->db = matrix(L->size_out, 1);
        L->mw = zero_matrix(L->size_in, L->size_out);
        L->vw = zero_matrix(L->size_in, L->size_out);
        L->mb = zero_matrix(L->size_out, 1);
        L->vb = zero_matrix(L->size_out, 1);
        double limit = sqrt(2.0 / L->size_in);
        random_matrix(L->w, L->size_in, L->size_out, limit);
        for(int k = 0; k < L->size_out; k++){
        	L->b[k][0] = 0;
		} 
        prev_size = this_size;
    }
    return net;
}

void free_network(NeuralNetworks net){
    for(int i = 0; i < net.layer; i++){
        Layer L = net.Layers[i];
        free_matrix(L.w, L.size_in);
        free_matrix(L.b, L.size_out);
        free_matrix(L.z, L.size_out);
        free_matrix(L.a, L.size_out);
        free_matrix(L.e, L.size_out);
        free_matrix(L.dw, L.size_in);
        free_matrix(L.db, L.size_out);
        free_matrix(L.mw, L.size_in);
        free_matrix(L.vw, L.size_in);
        free_matrix(L.mb, L.size_out);
        free_matrix(L.vb, L.size_out);
    }
    free(net.adam_t);
    free(net.Layers);
}

void feedforward_multilabel_train(NeuralNetworks net, double **input){
    int n = net.n_train;
    double **prev_a = input;
    for(int i = 0; i < net.layer; i++){
        Layer *L = &net.Layers[i];
        double **wT = transpose(L->w, L->size_in, L->size_out);
        double **z1 = matrix_multi(wT, L->size_out, L->size_in, prev_a, n);
        double **z2 = add_bias(z1, L->size_out, n, L->b);
        equal(L->z, L->size_out, n, z2);
        free_matrix(wT, L->size_out);
        free_matrix(z1, L->size_out);
        free_matrix(z2, L->size_out);
        if(i == net.layer - 1){
        	double **z3 = activation_matrix_Sigmoid(L->z, L->size_out, n); 
            equal(L->a, L->size_out, n, z3);
            free_matrix(z3, L->size_out); 
        } 
		else {
            double **act = activation_matrix_LeakyReLU(L->z, L->size_out, n);
            equal(L->a, L->size_out, n, act);
            free_matrix(act, L->size_out);
        }
        prev_a = L->a;
    }
}

void backpropagation_multilabel(NeuralNetworks net, double **input, double **y){
	int n = net.n_train;
	for(int i = net.layer - 1; i >= 0; i--){
		Layer *L = &net.Layers[i];
        if(i == net.layer - 1){
            for(int k = 0; k < L->size_out; k++){
            	for(int q = 0; q < n; q++){
            		L->e[k][q] = (L->a[k][q] - y[k][q]) / (L->size_out * n);
				}
			}                   
        } 
		else {
            Layer *Lnext = &net.Layers[i + 1];
            for(int j = 0; j < L->size_out; j++){
                for(int q = 0; q < n; q++){
                    double sum = 0;
                    for(int k = 0; k < Lnext->size_out; k++){
                    	sum += Lnext->e[k][q] * Lnext->w[j][k];
					}  
                    L->e[j][q] = gradLeakyReLU(L->z[j][q]) * sum;
                }
            }
        }
        double **prev_a = (i == 0) ? input : net.Layers[i - 1].a;
        double **eT = transpose(L->e, L->size_out, n);
        double **t  = matrix_multi(prev_a, L->size_in, n, eT, L->size_out);
        equal(L->dw, L->size_in, L->size_out, t);
        free_matrix(eT, n);
        free_matrix(t, L->size_in);
        for(int k = 0; k < L->size_out; k++){
            double s = 0;
            for(int q = 0; q < n; q++){
            	s += L->e[k][q];
			} 
            L->db[k][0] = s;
		}
	} 
} 

#define ADAM_BETA1 0.9 
#define ADAM_BETA2 0.999
#define ADAM_EPS   1e-8 

void adam_step(double **theta, double **grad, double **m, double **v, int rows, int cols, double alpha, int t){
    double beta1_t = pow(ADAM_BETA1, t);
    double beta2_t = pow(ADAM_BETA2, t);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            double g = grad[i][j];
            m[i][j] = ADAM_BETA1 * m[i][j] + (1 - ADAM_BETA1) * g;
            v[i][j] = ADAM_BETA2 * v[i][j] + (1 - ADAM_BETA2) * g * g;
            double m_hat = m[i][j] / (1 - beta1_t);
            double v_hat = v[i][j] / (1 - beta2_t);
            theta[i][j] -= alpha * m_hat / (sqrt(v_hat) + ADAM_EPS);
        }
    }
}

void update(NeuralNetworks net, double alpha){
    *net.adam_t += 1;
    for(int i = 0; i < net.layer; i++){
        Layer *L = &net.Layers[i];
        adam_step(L->w, L->dw, L->mw, L->vw, L->size_in, L->size_out, alpha, *net.adam_t);
        adam_step(L->b, L->db, L->mb, L->vb, L->size_out, 1, alpha, *net.adam_t);
    }
}

double BCE(double a, double b){
	return a * log(b) + (1 - a) * log(1 - b); 
} 

double cost(double **y, double **yhat, int m, int n){
	double sum = 0;
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			sum += BCE(y[i][j], yhat[i][j]); 
		} 
	} 
	return -sum / (m * n); 
} 


void train_multilabel(NeuralNetworks net, double **input, double **y, int output_dim, double alpha, int max_epoch, double target_cost){
	for(int epoch = 0; epoch < max_epoch; epoch++){
		feedforward_multilabel_train(net, input);
		backpropagation_multilabel(net, input, y);
		update(net, alpha);
		double loss = cost(y, net.Layers[net.layer - 1].a, output_dim, net.n_train);	
        if(epoch % 100 == 0){
        	printf("Epoch %5d | cost = %.8lf\n", epoch, loss);
		}
        if(loss < target_cost){
            printf("Hoi tu tai epoch %d | cost = %.8lf\n", epoch, loss);
            return;
        } 
	} 
	printf("Dat toi max_epoch, cost cuoi = %.8lf\n", cost(y, net.Layers[net.layer - 1].a, output_dim, net.n_train));
} 

double **predict_multilabel(NeuralNetworks net, double **input, int n_points, double threshold){
	double **prev_a = input;
    int owns_prev = 0; 
    for(int i = 0; i < net.layer; i++){
        Layer *L = &net.Layers[i];
        double **wT = transpose(L->w, L->size_in, L->size_out);
        double **z1 = matrix_multi(wT, L->size_out, L->size_in, prev_a, n_points);
        double **z2 = add_bias(z1, L->size_out, n_points, L->b);
        free_matrix(wT, L->size_out);
        free_matrix(z1, L->size_out);
        double **a_layer;
        if(i == net.layer - 1){
            a_layer = activation_matrix_Sigmoid(z2, L->size_out, n_points); 
			binary_vector(a_layer, L->size_out, n_points, threshold); 
        } 
		else{
            a_layer = activation_matrix_LeakyReLU(z2, L->size_out, n_points);
            free_matrix(z2, L->size_out);
        }
        if(owns_prev){
        	free_matrix(prev_a, L->size_in);
		} 
        prev_a = a_layer;
        owns_prev = 1;
    }
    return prev_a;
} 

int main(){
    srand((unsigned int)time(NULL));
    int n_train = 1000;
    int input_dim = 2;
	int output_dim = 3; 
    double *x_raw = (double*)malloc(n_train * sizeof(double));
    double *y_raw = (double*)malloc(n_train * sizeof(double));
    double **label = matrix(output_dim, n_train); 
    FILE *f = fopen("Multilabel.txt", "r");
    if(f == NULL){
        printf("Khong tim thay file du lieu\n");
    } 
	else{
        for(int i = 0; i < n_train; i++){
            fscanf(f, "%lf %lf %lf %lf %lf", &x_raw[i], &y_raw[i], &label[0][i], &label[1][i], &label[2][i]);
        }
        fclose(f);
    }
    double x_mi, x_ma, y_mi, y_ma; 
    scale_1d(x_raw, n_train, &x_mi, &x_ma);
    scale_1d(y_raw, n_train, &y_mi, &y_ma);
    double **input = matrix(input_dim, n_train);
    for(int i = 0; i < n_train; i++){
    	input[0][i] = x_raw[i];
    	input[1][i] = y_raw[i];  
	} 
    int hidden_sizes[] = {10, 10, 10};
    NeuralNetworks net = create_network(input_dim, hidden_sizes, 3, output_dim, n_train);
    train_multilabel(net, input, label, output_dim, 0.005, 50000, 0.005);
    int n_test = 10000;
    double *x_test_raw = (double*)malloc(n_test * sizeof(double)); 
    double *y_test_raw = (double*)malloc(n_test * sizeof(double));
    for(int i = 0; i < n_test; i++){
    	x_test_raw[i] = 1.0 * ((i % 100) + 1 - 50) / 50; 
	}  
	for(int i = 0; i < n_test; i++){ 
    	y_test_raw[i] = 1.0 * (i / 100 + 1 - 50) / 50; 
	}
    apply_scale_1d(x_test_raw, n_test, x_mi, x_ma);
    apply_scale_1d(y_test_raw, n_test, y_mi, y_ma);
    double **xy_test = matrix(input_dim, n_test);
    for(int i = 0; i < n_test; i++){
    	xy_test[0][i] = x_test_raw[i];
    	xy_test[1][i] = y_test_raw[i];
	}
    double **z_pred = predict_multilabel(net, xy_test, n_test, 0.5); 
    double *x_out = (double*)malloc(n_test * sizeof(double));
    double *y_out = (double*)malloc(n_test * sizeof(double));
    for(int i = 0; i < n_test; i++){
    	x_out[i] = x_test_raw[i];
    	y_out[i] = y_test_raw[i];
	}
    restore_1d(x_out, n_test, x_mi, x_ma);
    restore_1d(y_out, n_test, y_mi, y_ma);
    free_matrix(z_pred, output_dim);
    free_matrix(xy_test, input_dim);
    free_matrix(input, input_dim);
    free_matrix(label, output_dim); 
    free(x_raw); free(y_raw); 
    free(x_test_raw); free(y_test_raw);
	free(x_out); free(y_out); 
    free_network(net);
    return 0;
}
