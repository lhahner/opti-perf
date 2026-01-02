__kernel void adam_update(
    __global float*       param,   // p.data
    __global const float* grad,    // p.grad
    __global float*       m,       // st.m
    __global float*       v,       // st.v
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float bc1,              // 1 - beta1^t
    const float bc2,              // 1 - beta2^t
    const int   n                 // element count
){
    const int i = (int)get_global_id(0);
    if (i >= n) return;

    const float g  = grad[i];
    const float mi = beta1 * m[i] + (1.0f - beta1) * g;
    const float vi = beta2 * v[i] + (1.0f - beta2) * (g * g);

    m[i] = mi;
    v[i] = vi;

    const float mhat = mi / bc1;
    const float vhat = vi / bc2;

    param[i] -= lr * mhat / (sqrt(vhat) + eps);
}

