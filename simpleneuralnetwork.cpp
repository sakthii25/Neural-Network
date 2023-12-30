#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdlib>

#define float double
using namespace std;

class Optimizer{
    public:
    // here adam optimizer only implemented
    vector<float> m_w,v_w,m_b,v_b;
    float b1 = 0.9,b2 = 0.99,eps = 1e-8,lr = 0.01;
    Optimizer(int n){
        m_w.resize(n-1);v_w.resize(n-1);
        m_b.resize(n-1);v_b.resize(n-1);
    }
    vector<float> adam(int i,float w,float dw,float b,float db){

        m_w[i] = b1 * m_w[i] + (1-b1) * dw;
        m_b[i] = b1 * m_b[i] + (1-b1) * db;
        v_w[i] = b2 * v_w[i] + (1-b2) * dw * dw;
        v_b[i] = b2 * v_b[i] + (1-b2) * db * db;
        
        w -= lr * (m_w[i] / (sqrt(v_w[i]) + eps));
        b -= lr * (m_b[i] / (sqrt(v_b[i]) + eps));
        
        return {w,b};
    }
};
class Activations{
    public:
    // all the activation functions are implemented
    float activation(float z,string s){
        if(s == "linear") return z;
        if(s == "relu") return max(0.0,z);
        if(s == "tanh") return tanh(z);
        if(s == "sigmoid") return 1.0 / (1.0 + exp(-z));
        return 0.0;
    }
    // derivaties of activation function
    float d_activation(float z,string s){
        if (s == "linear") return 1.0;
        if (s == "relu") return (z > 0.0) ? 1.0 : 0.0;
        if (s == "tanh") {
            float tanh_z = tanh(z);
            return 1.0 - tanh_z * tanh_z;
        }
        if (s == "sigmoid") {
            float sigmoid_z = 1.0 / (1.0 + exp(-z));
            return sigmoid_z * (1.0 - sigmoid_z);
        }
        return 0.0;
    }
};
void initialize(vector<float>& w,vector<float>& b,vector<string>& layer){
    // Initialization of the weight and bias for each layer
    int n = w.size(),n_in = 1,n_out = 1;
    float stddev;
    srand(time(0));
    for(int i=0;i<n;++i){
        if(layer[i+1] == "linear"){
            stddev = 1;
        }
        else if(layer[i+1] == "relu"){
            stddev = sqrt(2.0 / n_in);
        }
        else{
            stddev = sqrt((2.0 / (n_in + n_out)));
        }
        w[i] = stddev * (static_cast<float>(rand()) / RAND_MAX);
        b[i] = stddev * (static_cast<float>(rand()) / RAND_MAX);
    }
}
float d_cost_function(float y,float pred,string type){
    // derivation of the cost function or loss function
    float cost = 0;
    if(type == "sigmoid"){
        pred = max(pred,1e-15);
        cost = ((1 - y) / (1 - pred)) - (y / pred);
    }
    else{
        cost = 2 * (pred - y);
    }
    return cost;
}
float loss_function(float y,float pred,string type){
    // loss function used to calculate the over all loss
    float loss = 0;
    if(type == "sigmoid"){
        loss = -((y*log(pred)) + ((1-y)*log(1-pred)));
    }
    else{
        loss = (pred - y) * (pred - y);
    }
    return loss;
}
float func(float x){
    //create a function for testcase generator
    //return (x > 500 ? 1 : 0);
    return (x * 2);
}
int main(){
    int n,N,epochs;
    n = 4;//No of layers
    N = 1000;// No of input test
    epochs = 1000;// No of epochs

    vector<string> layer = {"linear","relu","relu","linear"};
    //vector<string> layer = {"linear", "sigmoid", "sigmoid", "sigmoid"};
    vector<float> input(N);
    vector<float> output(N);

    for(int i=0;i<N;++i){
        input[i] = i+1;
        output[i] = func(input[i]);
    }
    
    vector<float> w(n-1),b(n-1);
    initialize(w,b,layer);
    Activations at;
    Optimizer opt(n);
    
    for(int epoch = 0;epoch < epochs; ++epoch){
        float total_loss = 0;
        vector<float> deltaw(n-1),deltab(n-1);
        
        for(int i=0;i<N;++i){

            float x = input[i],y = output[i];
            vector<float> z(n),a(n);
            z[0] = x;

            //forward pass
            for(int i=0;i+1<n;++i){
                if(!i) z[i+1] = z[i] * w[i] + b[i];
                else z[i+1] = a[i] * w[i] + b[i];
                a[i+1]  = at.activation(z[i+1],layer[i]);
            }

            //backward pass
            float cost = d_cost_function(y,a[n-1],layer[n-1]);
            float prev = at.d_activation(z[n-1],layer[n-1]);

            for(int i=n-2;i>=0;--i){
                deltaw[i] += cost * prev * z[i];
                deltab[i] += cost * prev;
                prev  = prev * w[i] * at.d_activation(z[i],layer[i]);
            }

            total_loss += loss_function(y,a[n-1],layer[n-1]);
        }

        for(int i=0;i+1<n;++i){
            float dw = deltaw[i]/N;
            float db = deltab[i]/N;
            auto res = opt.adam(i,w[i],dw,b[i],db);
            w[i] = res[0];
            b[i] = res[1];
        }

        if(epoch%100 == 0){
            cout << "Epoch[" << epoch << "/" << epochs << "]" << " Total Loss: " << total_loss/N << endl;
        }

    }

    //check 
    float x = 10;
    vector<float> z(n),a(n);
    z[0] = x;
    for(int i=0;i+1<n;++i){
        if(!i) z[i+1] = z[i] * w[i] + b[i];
        else z[i+1] = a[i] * w[i] + b[i];
        a[i+1]  = at.activation(z[i+1],layer[i]);
    }
    cout << "Actual value for Input " << x << ": " << func(x) << endl;
    cout << "Prediction for Input " << x << ": " << a[n-1] << endl;
}