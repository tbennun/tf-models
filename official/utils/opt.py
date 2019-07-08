import deep500 as d5
from deep500.frameworks import tensorflow as d5tf
import os
import os.path
import tensorflow as tf

import deep500.frameworks.tensorflow.custom_operators.tf as d5cop
import ctypes
def _custom_so_op(op, so_file, stateful, name):
    """ Registers a Deep500 Tensorflow operator from an existing .so file """

    # Load the compiled library into Tensorflow
    op_module = tf.load_op_library(so_file)
    op_func = getattr(op_module, 'tf_op' + op.name)
    op_grad_func = getattr(op_module, 'tf_op_grad' + op.name)
    
    # Create the deep500 custom op object
    lib = ctypes.CDLL(so_file)
    if not getattr(lib, 'create_new_op', False):
        raise ValueError('Invalid custom operator library file')
    lib.create_new_op.restype = ctypes.c_int64
    lib.is_cuda_supported.restype = ctypes.c_bool
    lib.report.restype = ctypes.c_int64

    return d5cop.TFCompiledOp(op, op_func, op_grad_func, lib)


class SoloDanceOptimizer(object):
    """ Enforces a sequential order on gradient exchange (by creating false 
        dependencies) and invokes a Deep500 custom operator based on MPI. """

    def __init__(self, optimizer: tf.train.Optimizer, comm_size: int):
        self.comm_size = comm_size
        self.optimizer = optimizer

        # Compile the operator
        opdesc = d5.compile_custom_cppop_inline('allreducef', _sallreduce,
                                                # Input tensor shapes (gradient, UNUSED last gradient op)
                                                [d5.tensordesc.runtime_shape(tf.float32), d5.tensordesc.runtime_shape(tf.float32)],
                                                # Output tensor shapes (reduced gradient)
                                                [d5.tensordesc.runtime_shape(tf.float32)],
                                                live_output=True)#, output_folder='/tmp')

        # If .so file exists, use cached file
        fname = os.environ['SOLO_SO'] if 'SOLO_SO' in os.environ else None
        if fname is not None and os.path.isfile(fname):
            self.compiled_op = _custom_so_op(opdesc, fname, True, None)
        else:
            self.compiled_op = d5tf.custom_op(opdesc, compile_only=True)

        self._handles = []

    def compute_gradients(self, *args, **kwargs):
        return self.optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, global_step):
        optimizer =  self.optimizer
        new_gvs = []
        last_g = None
        for grad, var in reversed(grads_and_vars):
            if grad is None:
                new_gvs.append((grad, var))
            else:
                if last_g is None:
                    last_g = grad
                    
            self.compiled_op.op.inputs = [d5tf.desc_from_tensor(grad), d5tf.desc_from_tensor(last_g)]
            self.compiled_op.op.outputs = [d5tf.desc_from_tensor(grad)]
            op, lib, handle = d5tf.custom_op(self.compiled_op, return_handle=True)
            self._handles.append((lib, handle))
            # Apply on gradient
            allreduced = op(grad, last_g)
            new_gvs.append((allreduced / (self.comm_size), var))
            last_g = allreduced

        return self.optimizer.apply_gradients(new_gvs, global_step)

_sallreduce = """
#include <deep500/deep500.h>
#include <mpi.h>

class allreducef : public deep500::CustomOperator {
protected:
  int m_len;
  int64_t m_totalbytes;
public:
  allreducef(int len) : m_len(len), m_totalbytes(0) { }
  virtual ~allreducef () { }
  virtual bool supports_cuda() { return false; } // TF copies the data automatically   
  virtual int64_t report(void *data) { return m_totalbytes; }

    void forward(const float *input, const float*, float *output) {
        MPI_Allreduce(input, output, m_len, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        m_totalbytes += m_len * sizeof(float);
    }

    void backward(const float *nextop_grad,
                  const float *fwd_input_tensor,
                  const float*,
                  const float *fwd_output_tensor,
                  float *input_tensor_grad,
                  float *) {
      // Do Nothing here
    }
};


D500_EXPORTED void *create_new_op(deep500::tensor_t *input_descriptors, int num_inputs,
                                  deep500::tensor_t *output_descriptors, int num_outputs) {
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (!is_init)
        MPI_Init(NULL, NULL);

    size_t totalsz = 1;
    for (int i = 0; i < input_descriptors[0].dims; ++i)
        totalsz *= input_descriptors[0].sizes[i];

    return new allreducef(totalsz);
}

D500_REGISTER_OP(allreducef);
"""
