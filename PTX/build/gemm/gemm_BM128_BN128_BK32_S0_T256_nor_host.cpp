# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(args: T.handle, arg_type_ids: T.handle("int32"), num_args: T.int32, out_ret_value: T.handle("void"), out_ret_tcode: T.handle("int32"), resource_handle: T.handle) -> T.int32:
        T.func_attr({"calling_conv": 1, "target": T.target({"keys": ["cpu"], "kind": "stackvm", "tag": ""}), "thread_extent": {}, "tir.is_entry_func": T.bool(True), "tma_descriptor_args": {}})
        assert num_args == 3, "main: num_args should be 3"
        assert not T.isnullptr(args), "main: TVMValue* arg pointer was NULL"
        assert not T.isnullptr(arg_type_ids), "main: int* type_codes was NULL"
        arg_type_ids_1 = T.decl_buffer((3,), "int32", data=arg_type_ids)
        A_handle_code: T.int32 = arg_type_ids_1[0]
        assert A_handle_code == 3 or A_handle_code == 13 or A_handle_code == 7 or A_handle_code == 4, "main: Expect arg[0] to be pointer"
        B_handle_code: T.int32 = arg_type_ids_1[1]
        assert B_handle_code == 3 or B_handle_code == 13 or B_handle_code == 7 or B_handle_code == 4, "main: Expect arg[1] to be pointer"
        C_handle_code: T.int32 = arg_type_ids_1[2]
        assert C_handle_code == 3 or C_handle_code == 13 or C_handle_code == 7 or C_handle_code == 4, "main: Expect arg[2] to be pointer"
        A_handle: T.handle = T.tvm_struct_get(args, 0, 12, "handle")
        B_handle: T.handle = T.tvm_struct_get(args, 1, 12, "handle")
        C_handle: T.handle = T.tvm_struct_get(args, 2, 12, "handle")
        assert not T.isnullptr(A_handle), "main.A_handle is expected to have non-NULL DLTensor* pointer"
        assert 2 == T.tvm_struct_get(A_handle, 0, 4, "int32"), "main.A_handle.ndim is expected to equal 2"
        main_A_handle_shape: T.handle("int64") = T.tvm_struct_get(A_handle, 0, 2, "handle")
        main_A_handle_shape_1 = T.decl_buffer((2,), "int64", data=main_A_handle_shape)
        main_A_handle_strides: T.handle("int64") = T.tvm_struct_get(A_handle, 0, 3, "handle")
        main_A_handle_strides_1 = T.decl_buffer((0,), "int64", data=main_A_handle_strides)
        dev_id: T.int32 = T.tvm_struct_get(A_handle, 0, 9, "int32")
        A: T.handle("float16", "global") = T.tvm_struct_get(A_handle, 0, 1, "handle")
        T.attr(A, "storage_alignment", 64)
        assert not T.isnullptr(B_handle), "main.B_handle is expected to have non-NULL DLTensor* pointer"
        assert 2 == T.tvm_struct_get(B_handle, 0, 4, "int32"), "main.B_handle.ndim is expected to equal 2"
        main_B_handle_shape: T.handle("int64") = T.tvm_struct_get(B_handle, 0, 2, "handle")
        main_B_handle_shape_1 = T.decl_buffer((2,), "int64", data=main_B_handle_shape)
        main_B_handle_strides: T.handle("int64") = T.tvm_struct_get(B_handle, 0, 3, "handle")
        main_B_handle_strides_1 = T.decl_buffer((0,), "int64", data=main_B_handle_strides)
        B: T.handle("float16", "global") = T.tvm_struct_get(B_handle, 0, 1, "handle")
        T.attr(B, "storage_alignment", 64)
        assert not T.isnullptr(C_handle), "main.C_handle is expected to have non-NULL DLTensor* pointer"
        assert 2 == T.tvm_struct_get(C_handle, 0, 4, "int32"), "main.C_handle.ndim is expected to equal 2"
        main_C_handle_shape: T.handle("int64") = T.tvm_struct_get(C_handle, 0, 2, "handle")
        main_C_handle_shape_1 = T.decl_buffer((2,), "int64", data=main_C_handle_shape)
        main_C_handle_strides: T.handle("int64") = T.tvm_struct_get(C_handle, 0, 3, "handle")
        main_C_handle_strides_1 = T.decl_buffer((0,), "int64", data=main_C_handle_strides)
        C: T.handle("float16", "global") = T.tvm_struct_get(C_handle, 0, 1, "handle")
        T.attr(C, "storage_alignment", 64)
        T.attr("default", "device_id", dev_id)
        T.attr("default", "device_type", 2)
        assert T.tvm_struct_get(A_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(A_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(A_handle, 0, 7, "uint16") == T.uint16(1), "main.A_handle.dtype is expected to be float16"
        assert T.Cast("int32", main_A_handle_shape_1[0]) == 4096, "Argument main.A_handle.shape[0] has an unsatisfied constraint: 4096 == T.Cast(\"int32\", main_A_handle_shape[0])"
        assert T.Cast("int32", main_A_handle_shape_1[1]) == 4096, "Argument main.A_handle.shape[1] has an unsatisfied constraint: 4096 == T.Cast(\"int32\", main_A_handle_shape[1])"
        if not T.isnullptr(main_A_handle_strides):
            assert 1 == T.Cast("int32", main_A_handle_strides_1[1]) and 4096 == T.Cast("int32", main_A_handle_strides_1[0]), "main.A_handle.strides: expected to be compact array"
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(A_handle, 0, 8, "uint64"), "Argument main.A_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(A_handle, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(A_handle, 0, 10, "int32") == 2, "Argument main.A_handle.device_type has an unsatisfied constraint: 2 == T.tvm_struct_get(A_handle, 0, 10, \"int32\")"
        assert not T.isnullptr(A), "main.A_handle is expected to have non-NULL data pointer"
        assert T.tvm_struct_get(B_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(B_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(B_handle, 0, 7, "uint16") == T.uint16(1), "main.B_handle.dtype is expected to be float16"
        assert T.Cast("int32", main_B_handle_shape_1[0]) == 4096, "Argument main.B_handle.shape[0] has an unsatisfied constraint: 4096 == T.Cast(\"int32\", main_B_handle_shape[0])"
        assert T.Cast("int32", main_B_handle_shape_1[1]) == 4096, "Argument main.B_handle.shape[1] has an unsatisfied constraint: 4096 == T.Cast(\"int32\", main_B_handle_shape[1])"
        if not T.isnullptr(main_B_handle_strides):
            assert 1 == T.Cast("int32", main_B_handle_strides_1[1]) and 4096 == T.Cast("int32", main_B_handle_strides_1[0]), "main.B_handle.strides: expected to be compact array"
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(B_handle, 0, 8, "uint64"), "Argument main.B_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(B_handle, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(B_handle, 0, 10, "int32") == 2, "Argument main.B_handle.device_type has an unsatisfied constraint: 2 == T.tvm_struct_get(B_handle, 0, 10, \"int32\")"
        assert dev_id == T.tvm_struct_get(B_handle, 0, 9, "int32"), "Argument main.B_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(B_handle, 0, 9, \"int32\")"
        assert not T.isnullptr(B), "main.B_handle is expected to have non-NULL data pointer"
        assert T.tvm_struct_get(C_handle, 0, 5, "uint8") == T.uint8(2) and T.tvm_struct_get(C_handle, 0, 6, "uint8") == T.uint8(16) and T.tvm_struct_get(C_handle, 0, 7, "uint16") == T.uint16(1), "main.C_handle.dtype is expected to be float16"
        assert T.Cast("int32", main_C_handle_shape_1[0]) == 4096, "Argument main.C_handle.shape[0] has an unsatisfied constraint: 4096 == T.Cast(\"int32\", main_C_handle_shape[0])"
        assert T.Cast("int32", main_C_handle_shape_1[1]) == 4096, "Argument main.C_handle.shape[1] has an unsatisfied constraint: 4096 == T.Cast(\"int32\", main_C_handle_shape[1])"
        if not T.isnullptr(main_C_handle_strides):
            assert 1 == T.Cast("int32", main_C_handle_strides_1[1]) and 4096 == T.Cast("int32", main_C_handle_strides_1[0]), "main.C_handle.strides: expected to be compact array"
            T.evaluate(0)
        assert T.uint64(0) == T.tvm_struct_get(C_handle, 0, 8, "uint64"), "Argument main.C_handle.byte_offset has an unsatisfied constraint: T.uint64(0) == T.tvm_struct_get(C_handle, 0, 8, \"uint64\")"
        assert T.tvm_struct_get(C_handle, 0, 10, "int32") == 2, "Argument main.C_handle.device_type has an unsatisfied constraint: 2 == T.tvm_struct_get(C_handle, 0, 10, \"int32\")"
        assert dev_id == T.tvm_struct_get(C_handle, 0, 9, "int32"), "Argument main.C_handle.device_id has an unsatisfied constraint: dev_id == T.tvm_struct_get(C_handle, 0, 9, \"int32\")"
        assert not T.isnullptr(C), "main.C_handle is expected to have non-NULL data pointer"
        A_1 = T.decl_buffer((4096, 4096), "float16", data=A)
        B_1 = T.decl_buffer((4096, 4096), "float16", data=B)
        C_1 = T.decl_buffer((4096, 4096), "float16", data=C)
        assert T.FloorMod(4096, 8) == 0, "A: Vectorize dimension in buffer must be divisible by 8"
        assert T.FloorMod(4096, 8) == 0, "B: Vectorize dimension in buffer must be divisible by 8"
        assert T.FloorMod(4096, 8) == 0, "C: Vectorize dimension in buffer must be divisible by 8"
        T.call_packed("__tvm_set_device", 2, dev_id)
        with T.attr(0, "compute_scope", "main_compute_"):
            T.call_packed("main_kernel", A, B, C, 32, 32, 256, 1, 1, 16384)
        return 0