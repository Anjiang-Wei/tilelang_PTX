# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(args: T.handle, arg_type_ids: T.handle("int32"), num_args: T.int32, out_ret_value: T.handle("void"), out_ret_tcode: T.handle("int32"), resource_handle: T.handle) -> T.int32:
        T.func_attr({"calling_conv": 1, "target": T.target({"keys": ["cpu"], "kind": "c", "tag": ""}), "thread_extent": {}, "tir.is_entry_func": True, "tma_descriptor_args": {}})
        assert num_args == 3, "main: num_args should be 3"
        assert not T.isnullptr(args), "main: TVMValue* arg pointer was NULL"
        assert not T.isnullptr(arg_type_ids), "main: int* type_codes was NULL"
        arg_type_ids_1 = T.decl_buffer((3,), "int32", data=arg_type_ids)
        data_handle_code: T.int32 = arg_type_ids_1[0]
        assert data_handle_code == 0 or data_handle_code == 4 or data_handle_code == 7 or data_handle_code >= 64, "main: Expect arg[0] to be pointer"
        kernel_handle_code: T.int32 = arg_type_ids_1[1]
        assert kernel_handle_code == 0 or kernel_handle_code == 4 or kernel_handle_code == 7 or kernel_handle_code >= 64, "main: Expect arg[1] to be pointer"
        out_handle_code: T.int32 = arg_type_ids_1[2]
        assert out_handle_code == 0 or out_handle_code == 4 or out_handle_code == 7 or out_handle_code >= 64, "main: Expect arg[2] to be pointer"
        data_handle: T.handle = T.tvm_struct_get(args, 0, 12, "handle")
        kernel_handle: T.handle = T.tvm_struct_get(args, 1, 12, "handle")
        out_handle: T.handle = T.tvm_struct_get(args, 2, 12, "handle")
        assert not T.isnullptr(data_handle), "main.data_handle is expected to have non-NULL DLTensor* pointer"
        assert 4 == T.tvm_struct_get(data_handle, 0, 4, "int32"), "main.data_handle.ndim is expected to equal 4"
        main_data_handle_shape: T.handle("int64") = T.tvm_struct_get(data_handle, 0, 2, "handle")
        main_data_handle_shape_1 = T.decl_buffer((4,), "int64", data=main_data_handle_shape)
        main_data_handle_strides: T.handle("int64") = T.tvm_struct_get(data_handle, 0, 3, "handle")
        main_data_handle_strides_1 = T.decl_buffer((4,), "int64", data=main_data_handle_strides)
        dev_id: T.int32 = T.tvm_struct_get(data_handle, 0, 9, "int32")
        data: T.handle("float16", "global") = T.tvm_struct_get(data_handle, 0, 1, "handle")
        T.attr(data, "storage_alignment", 64)
        assert not T.isnullptr(kernel_handle), "main.kernel_handle is expected to have non-NULL DLTensor* pointer"
        assert 4 == T.tvm_struct_get(kernel_handle, 0, 4, "int32"), "main.kernel_handle.ndim is expected to equal 4"
        main_kernel_handle_shape: T.handle("int64") = T.tvm_struct_get(kernel_handle, 0, 2, "handle")
        main_kernel_handle_shape_1 = T.decl_buffer((4,), "int64", data=main_kernel_handle_shape)
        main_kernel_handle_strides: T.handle("int64") = T.tvm_struct_get(kernel_handle, 0, 3, "handle")
        main_kernel_handle_strides_1 = T.decl_buffer((4,), "int64", data=main_kernel_handle_strides)
        kernel_flat: T.handle("float16", "global") = T.tvm_struct_get(kernel_handle, 0, 1, "handle")
        T.attr(kernel_flat, "storage_alignment", 64)
        assert not T.isnullptr(out_handle), "main.out_handle is expected to have non-NULL DLTensor* pointer"
        assert 4 == T.tvm_struct_get(out_handle, 0, 4, "int32"), "main.out_handle.ndim is expected to equal 4"
        main_out_handle_shape: T.handle("int64") = T.tvm_struct_get(out_handle, 0, 2, "handle")
        main_out_handle_shape_1 = T.decl_buffer((4,), "int64", data=main_out_handle_shape)
        main_out_handle_strides: T.handle("int64") = T.tvm_struct_get(out_handle, 0, 3, "handle")
        main_out_handle_strides_1 = T.decl_buffer((4,), "int64", data=main_out_handle_strides)
        out_flat: T.handle("float16", "global") = T.tvm_struct_get(out_handle, 0, 1, "handle")
        T.attr(out_flat, "storage_alignment", 64)
        T.attr("default", "device_id", dev_id)
        T.attr("default", "device_type", 2)
        assert T.tvm_struct_get(data_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(data_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(data_handle, 0, 7, "uint16") == T.uint16(1), "main.data_handle.dtype is expected to be float16"
        assert T.Cast("int32", main_data_handle_shape_1[0]) == 128, "Argument main.data_handle.shape[0] has an unsatisfied constraint: 128 == T.Cast(\"int32\", main_data_handle_shape[0])"
        assert T.Cast("int32", main_data_handle_shape_1[1]) == 64, "Argument main.data_handle.shape[1] has an unsatisfied constraint: 64 == T.Cast(\"int32\", main_data_handle_shape[1])"
        assert T.Cast("int32", main_data_handle_shape_1[2]) == 64, "Argument main.data_handle.shape[2] has an unsatisfied constraint: 64 == T.Cast(\"int32\", main_data_handle_shape[2])"
        assert T.Cast("int32", main_data_handle_shape_1[3]) == 128, "Argument main.data_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\"int32\", main_data_handle_shape[3])"
        assert T.if_then_else(T.isnullptr(main_data_handle_strides), 1, T.Cast("int32", main_data_handle_strides_1[3])) == 1, "Argument main.data_handle.strides[3] has an unsatisfied constraint: 1 == T.if_then_else(T.isnullptr(main_data_handle_strides), 1, T.Cast(\"int32\", main_data_handle_strides_1[3]))"
        assert T.if_then_else(T.isnullptr(main_data_handle_strides), T.Cast("int32", main_data_handle_shape_1[3]), T.Cast("int32", main_data_handle_strides_1[2])) == 128, "Argument main.data_handle.strides[2] has an unsatisfied constraint: 128 == T.if_then_else(T.isnullptr(main_data_handle_strides), T.Cast(\"int32\", main_data_handle_shape[3]), T.Cast(\"int32\", main_data_handle_strides_1[2]))"
        assert T.if_then_else(T.isnullptr(main_data_handle_strides), T.Cast("int32", main_data_handle_shape_1[3]) * T.Cast("int32", main_data_handle_shape_1[2]), T.Cast("int32", main_data_handle_strides_1[1])) == 8192, "Argument main.data_handle.strides[1] has an unsatisfied constraint: 8192 == T.if_then_else(T.isnullptr(main_data_handle_strides), T.Cast(\"int32\", main_data_handle_shape[3]) * T.Cast(\"int32\", main_data_handle_shape[2]), T.Cast(\"int32\", main_data_handle_strides_1[1]))"
        assert T.if_then_else(T.isnullptr(main_data_handle_strides), T.Cast("int32", main_data_handle_shape_1[3]) * T.Cast("int32", main_data_handle_shape_1[2]) * T.Cast("int32", main_data_handle_shape_1[1]), T.Cast("int32", main_data_handle_strides_1[0])) == 524288, "Argument main.data_handle.strides[0] has an unsatisfied constraint: 524288 == T.if_then_else(T.isnullptr(main_data_handle_strides), T.Cast(\"int32\", main_data_handle_shape[3]) * T.Cast(\"int32\", main_data_handle_shape[2]) * T.Cast(\"int32\", main_data_handle_shape[1]), T.Cast(\"int32\", main_data_handle_strides_1[0]))"
        assert T.uint64(0) == T.tvm_struct_get(data_handle, 0, 8, "uint64"), "Argument main.data_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(data_handle, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(data_handle, 0, 10, "int32") == 2, "Argument main.data_handle.device_type has an unsatisfied constraint: 2 == T.tvm_struct_get(data_handle, 0, 10, \"int32\")"
        assert not T.isnullptr(data), "main.data_handle is expected to have non-NULL data pointer"
        assert T.tvm_struct_get(kernel_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(kernel_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(kernel_handle, 0, 7, "uint16") == T.uint16(1), "main.kernel_handle.dtype is expected to be float16"
        assert T.Cast("int32", main_kernel_handle_shape_1[0]) == 3, "Argument main.kernel_handle.shape[0] has an unsatisfied constraint: 3 == T.Cast(\"int32\", main_kernel_handle_shape[0])"
        assert T.Cast("int32", main_kernel_handle_shape_1[1]) == 3, "Argument main.kernel_handle.shape[1] has an unsatisfied constraint: 3 == T.Cast(\"int32\", main_kernel_handle_shape[1])"
        assert T.Cast("int32", main_kernel_handle_shape_1[2]) == 128, "Argument main.kernel_handle.shape[2] has an unsatisfied constraint: 128 == T.Cast(\"int32\", main_kernel_handle_shape[2])"
        assert T.Cast("int32", main_kernel_handle_shape_1[3]) == 128, "Argument main.kernel_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\"int32\", main_kernel_handle_shape[3])"
        assert T.if_then_else(T.isnullptr(main_kernel_handle_strides), 1, T.Cast("int32", main_kernel_handle_strides_1[3])) == 1, "Argument main.kernel_handle.strides[3] has an unsatisfied constraint: 1 == T.if_then_else(T.isnullptr(main_kernel_handle_strides), 1, T.Cast(\"int32\", main_kernel_handle_strides_1[3]))"
        assert T.if_then_else(T.isnullptr(main_kernel_handle_strides), T.Cast("int32", main_kernel_handle_shape_1[3]), T.Cast("int32", main_kernel_handle_strides_1[2])) == 128, "Argument main.kernel_handle.strides[2] has an unsatisfied constraint: 128 == T.if_then_else(T.isnullptr(main_kernel_handle_strides), T.Cast(\"int32\", main_kernel_handle_shape[3]), T.Cast(\"int32\", main_kernel_handle_strides_1[2]))"
        assert T.if_then_else(T.isnullptr(main_kernel_handle_strides), T.Cast("int32", main_kernel_handle_shape_1[3]) * T.Cast("int32", main_kernel_handle_shape_1[2]), T.Cast("int32", main_kernel_handle_strides_1[1])) == 16384, "Argument main.kernel_handle.strides[1] has an unsatisfied constraint: 16384 == T.if_then_else(T.isnullptr(main_kernel_handle_strides), T.Cast(\"int32\", main_kernel_handle_shape[3]) * T.Cast(\"int32\", main_kernel_handle_shape[2]), T.Cast(\"int32\", main_kernel_handle_strides_1[1]))"
        assert T.if_then_else(T.isnullptr(main_kernel_handle_strides), T.Cast("int32", main_kernel_handle_shape_1[3]) * T.Cast("int32", main_kernel_handle_shape_1[2]) * T.Cast("int32", main_kernel_handle_shape_1[1]), T.Cast("int32", main_kernel_handle_strides_1[0])) == 49152, "Argument main.kernel_handle.strides[0] has an unsatisfied constraint: 49152 == T.if_then_else(T.isnullptr(main_kernel_handle_strides), T.Cast(\"int32\", main_kernel_handle_shape[3]) * T.Cast(\"int32\", main_kernel_handle_shape[2]) * T.Cast(\"int32\", main_kernel_handle_shape[1]), T.Cast(\"int32\", main_kernel_handle_strides_1[0]))"
        assert T.uint64(0) == T.tvm_struct_get(kernel_handle, 0, 8, "uint64"), "Argument main.kernel_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(kernel_handle, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(kernel_handle, 0, 10, "int32") == 2, "Argument main.kernel_handle.device_type has an unsatisfied constraint: 2 == T.tvm_struct_get(kernel_handle, 0, 10, \"int32\")"
        assert dev_id == T.tvm_struct_get(kernel_handle, 0, 9, "int32"), "Argument main.kernel_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(kernel_handle, 0, 9, \"int32\")"
        assert not T.isnullptr(kernel_flat), "main.kernel_handle is expected to have non-NULL data pointer"
        assert T.tvm_struct_get(out_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(out_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(out_handle, 0, 7, "uint16") == T.uint16(1), "main.out_handle.dtype is expected to be float16"
        assert T.Cast("int32", main_out_handle_shape_1[0]) == 128, "Argument main.out_handle.shape[0] has an unsatisfied constraint: 128 == T.Cast(\"int32\", main_out_handle_shape[0])"
        assert T.Cast("int32", main_out_handle_shape_1[1]) == 64, "Argument main.out_handle.shape[1] has an unsatisfied constraint: 64 == T.Cast(\"int32\", main_out_handle_shape[1])"
        assert T.Cast("int32", main_out_handle_shape_1[2]) == 64, "Argument main.out_handle.shape[2] has an unsatisfied constraint: 64 == T.Cast(\"int32\", main_out_handle_shape[2])"
        assert T.Cast("int32", main_out_handle_shape_1[3]) == 128, "Argument main.out_handle.shape[3] has an unsatisfied constraint: 128 == T.Cast(\"int32\", main_out_handle_shape[3])"
        assert T.if_then_else(T.isnullptr(main_out_handle_strides), 1, T.Cast("int32", main_out_handle_strides_1[3])) == 1, "Argument main.out_handle.strides[3] has an unsatisfied constraint: 1 == T.if_then_else(T.isnullptr(main_out_handle_strides), 1, T.Cast(\"int32\", main_out_handle_strides_1[3]))"
        assert T.if_then_else(T.isnullptr(main_out_handle_strides), T.Cast("int32", main_out_handle_shape_1[3]), T.Cast("int32", main_out_handle_strides_1[2])) == 128, "Argument main.out_handle.strides[2] has an unsatisfied constraint: 128 == T.if_then_else(T.isnullptr(main_out_handle_strides), T.Cast(\"int32\", main_out_handle_shape[3]), T.Cast(\"int32\", main_out_handle_strides_1[2]))"
        assert T.if_then_else(T.isnullptr(main_out_handle_strides), T.Cast("int32", main_out_handle_shape_1[3]) * T.Cast("int32", main_out_handle_shape_1[2]), T.Cast("int32", main_out_handle_strides_1[1])) == 8192, "Argument main.out_handle.strides[1] has an unsatisfied constraint: 8192 == T.if_then_else(T.isnullptr(main_out_handle_strides), T.Cast(\"int32\", main_out_handle_shape[3]) * T.Cast(\"int32\", main_out_handle_shape[2]), T.Cast(\"int32\", main_out_handle_strides_1[1]))"
        assert T.if_then_else(T.isnullptr(main_out_handle_strides), T.Cast("int32", main_out_handle_shape_1[3]) * T.Cast("int32", main_out_handle_shape_1[2]) * T.Cast("int32", main_out_handle_shape_1[1]), T.Cast("int32", main_out_handle_strides_1[0])) == 524288, "Argument main.out_handle.strides[0] has an unsatisfied constraint: 524288 == T.if_then_else(T.isnullptr(main_out_handle_strides), T.Cast(\"int32\", main_out_handle_shape[3]) * T.Cast(\"int32\", main_out_handle_shape[2]) * T.Cast(\"int32\", main_out_handle_shape[1]), T.Cast(\"int32\", main_out_handle_strides_1[0]))"
        assert T.uint64(0) == T.tvm_struct_get(out_handle, 0, 8, "uint64"), "Argument main.out_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(out_handle, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(out_handle, 0, 10, "int32") == 2, "Argument main.out_handle.device_type has an unsatisfied constraint: 2 == T.tvm_struct_get(out_handle, 0, 10, \"int32\")"
        assert dev_id == T.tvm_struct_get(out_handle, 0, 9, "int32"), "Argument main.out_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(out_handle, 0, 9, \"int32\")"
        assert not T.isnullptr(out_flat), "main.out_handle is expected to have non-NULL data pointer"
        data_1 = T.decl_buffer((128, 64, 64, 128), "float16", data=data, strides=(524288, 8192, 128, 1))
        kernel = T.decl_buffer((3, 3, 128, 128), "float16", data=kernel_flat, strides=(49152, 16384, 128, 1))
        out = T.decl_buffer((128, 64, 64, 128), "float16", data=out_flat, strides=(524288, 8192, 128, 1))
        assert T.FloorMod(128, 8) == 0, "data: Vectorize dimension in buffer must be divisible by 8"
        assert T.FloorMod(128, 8) == 0, "kernel: Vectorize dimension in buffer must be divisible by 8"
        assert T.FloorMod(128, 8) == 0, "out: Vectorize dimension in buffer must be divisible by 8"
        T.call_packed("__tvm_set_device", 2, dev_id)
        with T.attr(0, "compute_scope", "main_compute_"):
            T.call_packed("main_kernel", data, kernel_flat, out_flat, 1, 8192, 128, 1, 1, 24576)
        return 0