// Copyright 2019 ETH Zurich and University of Bologna.
//
// Copyright and related rights are licensed under the Solderpad Hardware
// License, Version 0.51 (the "License"); you may not use this file except in
// compliance with the License. You may obtain a copy of the License at
// http://solderpad.org/licenses/SHL-0.51. Unless required by applicable law
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
//
// SPDX-License-Identifier: SHL-0.51

// Author: Stefan Mach <smach@iis.ee.ethz.ch>

module spatz_fpu #(
  // FPU configuration
  parameter fpnew_pkg::fpu_features_t       Features       = fpnew_pkg::RV64D_Xsflt,
  parameter fpnew_pkg::fpu_implementation_t Implementation = fpnew_pkg::DEFAULT_NOREGS,
  // DivSqrtSel chooses among PULP, TH32, or THMULTI (see documentation and fpnew_pkg.sv for further details)
  parameter fpnew_pkg::divsqrt_unit_t       DivSqrtSel     = fpnew_pkg::THMULTI,
  parameter type                            TagType        = logic,
  parameter logic                           TrueSIMDClass  = 1'b0,
  parameter logic                           EnableSIMDMask = 1'b0,
  parameter logic                           CompressedVecCmpResult = 1'b0, // conceived for RV32FD cores
  parameter fpnew_pkg::rsr_impl_t           StochasticRndImplementation = fpnew_pkg::DEFAULT_NO_RSR,
  // Do not change
  localparam int unsigned NumLanes     = fpnew_pkg::max_num_lanes(Features.Width, Features.FpFmtMask, Features.EnableVectors),
  localparam type         MaskType     = logic [NumLanes-1:0],
  localparam int unsigned WIDTH        = Features.Width,
  localparam int unsigned NUM_OPERANDS = 3
) (
  input logic                               clk_i,
  input logic                               rst_ni,
  input logic [31:0]                        hart_id_i,
  // Input signals
  input logic [NUM_OPERANDS-1:0][WIDTH-1:0] operands_i,
  input fpnew_pkg::roundmode_e              rnd_mode_i,
  input fpnew_pkg::operation_e              op_i,
  input logic                               op_mod_i,
  input fpnew_pkg::fp_format_e              src_fmt_i,
  input fpnew_pkg::fp_format_e              dst_fmt_i,
  input fpnew_pkg::int_format_e             int_fmt_i,
  input logic                               vectorial_op_i,
  input TagType                             tag_i,
  input MaskType                            simd_mask_i,
  // Input Handshake
  input  logic                              in_valid_i,
  output logic                              in_ready_o,
  input  logic                              flush_i,
  // Output signals
  output logic [WIDTH-1:0]                  result_o,
  output fpnew_pkg::status_t                status_o,
  output TagType                            tag_o,
  // Output handshake
  output logic                              out_valid_o,
  input  logic                              out_ready_i,
  // Indication of valid data in flight
  output logic                              busy_o
);
  // Include FF
  `include "common_cells/registers.svh"
  localparam int unsigned NUM_OPGROUPS = fpnew_pkg::NUM_OPGROUPS;
  localparam int unsigned NUM_FORMATS  = fpnew_pkg::NUM_FP_FORMATS;

  // ----------------
  // Type Definition
  // ----------------
  typedef struct packed {
    logic [WIDTH-1:0]   result;
    fpnew_pkg::status_t status;
    TagType             tag;
  } output_t;

  typedef enum logic [2:0] { NL_IDLE, NL_FPU_ISSUE_0,NL_FPU_ISSUE_1,NL_SUM_EXP, NL_WAIT } nl_phase_e;
  typedef enum logic [2:0] { EXPS, COSHS } nl_op_e;

  TagType fconv_tag, fadd_tag;
  logic nl_concatenate;
  logic nl_opmode_conv, nl_opmode_add;
  logic nl_wr_en, is_last_uop;
  
  
  fpnew_pkg::operation_e nl_op_conv, nl_op_add, nl_op;
  fpnew_pkg::roundmode_e nl_rnd_conv, nl_rnd_add, nl_rnd;
  assign nl_concatenate = tag_i.nl;

  logic [NUM_OPERANDS-1:0][WIDTH-1:0]  operands_fconv, operands_add;
  
  // Handshake signals for the blocks
  logic [NUM_OPGROUPS-1:0] opgrp_in_ready, opgrp_out_valid, concatenate_out_valid, opgrp_out_ready, opgrp_ext, opgrp_busy, out_opgrp_ready;
  logic [WIDTH-1:0] nl_intermediate;
  `FFL(nl_intermediate, opgrp_outputs[fpnew_pkg::CONV].result, nl_wr_en, 1'b0)
  
  output_t [NUM_OPGROUPS-1:0] opgrp_outputs;

  logic [NUM_FORMATS-1:0][NUM_OPERANDS-1:0] is_boxed;
  logic [NUM_OPGROUPS-1:0] concatenate_in_valid;
  logic add_phase_q, add_phase_d;
  `FF(add_phase_q, add_phase_d, 1'b0)

  always_comb begin : select_inputs
    operands_fconv = '0;
    operands_add   = '0;
    nl_opmode_conv = 1'b0;
    nl_opmode_add  = 1'b0;
    nl_op_conv     = fpnew_pkg::F2I;
    nl_op_add      = fpnew_pkg::ADD;
    nl_rnd_conv    = fpnew_pkg::RTZ;
    nl_rnd_add     = fpnew_pkg::RNE;
    fconv_tag      = '0;
    fadd_tag       = '0;
    out_opgrp_ready = opgrp_out_ready;
    nl_wr_en       = 1'b0;
    add_phase_d    = add_phase_q;

    unique case(tag_i.nl_op_sel)
      EXPS: begin
        if (nl_concatenate && opgrp_out_valid[fpnew_pkg::ADDMUL]) begin
          operands_fconv[0]                   = opgrp_outputs[0].result;
          operands_fconv[1]                   = '0;
          operands_fconv[2]                   = '0;   
          nl_opmode_conv                      = 1'b0;
          nl_op_conv                          = fpnew_pkg::F2I;
          nl_rnd_conv                         = fpnew_pkg::RTZ;
          fconv_tag                           =  opgrp_outputs[fpnew_pkg::ADDMUL].tag;
          out_opgrp_ready[0]                  = opgrp_in_ready[3] ? 'b1 : 'b0; 
          //out_opgrp_ready[NUM_OPGROUPS-1 :1]  = opgrp_out_ready[NUM_OPGROUPS-1 :1]; 
        end 
      end
      COSHS: begin
        if (nl_concatenate && opgrp_out_valid[fpnew_pkg::ADDMUL] && opgrp_outputs[0].tag.last_phase != 1'b1) begin
          operands_fconv[0]                   = opgrp_outputs[0].result;
          operands_fconv[1]                   = '0;
          operands_fconv[2]                   = '0;   
          nl_opmode_conv                      = 1'b0;
          nl_op_conv                          = fpnew_pkg::F2I;
          nl_rnd_conv                         = fpnew_pkg::RTZ;
          fconv_tag                           =  opgrp_outputs[fpnew_pkg::ADDMUL].tag;
          out_opgrp_ready[0]                  = opgrp_in_ready[3] ? 'b1 : 'b0; 
          add_phase_d                         = 1'b1;
          
        end 
        if (nl_concatenate && opgrp_out_valid[fpnew_pkg::ADDMUL] && opgrp_outputs[0].tag.last_phase != 1'b0) begin
          out_opgrp_ready[0]                  = 1'b1;
          is_last_uop                         = 1'b1;
        end
      
        if (nl_concatenate && opgrp_out_valid[fpnew_pkg::CONV] && opgrp_outputs[3].tag.nl_phase == NL_FPU_ISSUE_1) begin
          operands_add[0]                     = nl_intermediate;
          operands_add[1]                     = opgrp_outputs[fpnew_pkg::CONV].result;
          operands_add[2]                     = nl_intermediate;   
          nl_opmode_add                       = 1'b0;
          nl_op_add                           = fpnew_pkg::ADD;
          nl_rnd_add                          = fpnew_pkg::RNE;
          fadd_tag                            = opgrp_outputs[fpnew_pkg::CONV].tag;
          out_opgrp_ready[3]                  = opgrp_in_ready[0] ? 'b1 : 'b0; 
        end

        if (nl_concatenate && opgrp_out_valid[fpnew_pkg::CONV] && opgrp_outputs[3].tag.nl_phase== NL_FPU_ISSUE_0) begin
          nl_wr_en = 1'b1;
          out_opgrp_ready[3]                  = 1'b1; 
          fadd_tag.last_phase                 = 1'b1;
        end
      end
    endcase
  end
  // -----------
  // Input Side
  // -----------
  assign in_ready_o = in_valid_i & opgrp_in_ready[fpnew_pkg::get_opgroup(op_i)];

  // NaN-boxing check
  for (genvar fmt = 0; fmt < int'(NUM_FORMATS); fmt++) begin : gen_nanbox_check
    localparam int unsigned FP_WIDTH = fpnew_pkg::fp_width(fpnew_pkg::fp_format_e'(fmt));
    // NaN boxing is only generated if it's enabled and needed
    if (Features.EnableNanBox && (FP_WIDTH < WIDTH)) begin : check
      for (genvar op = 0; op < int'(NUM_OPERANDS); op++) begin : operands
        assign is_boxed[fmt][op] = (!vectorial_op_i)
                                   ? operands_i[op][WIDTH-1:FP_WIDTH] == '1
                                   : 1'b1;
      end
    end else begin : no_check
      assign is_boxed[fmt] = '1;
    end
  end

  // Filter out the mask if not used
  MaskType simd_mask;
  assign simd_mask = simd_mask_i | ~{NumLanes{EnableSIMDMask}};

  // -------------------------
  // Generate Operation Blocks
  // -------------------------
  for (genvar opgrp = 0; opgrp < int'(NUM_OPGROUPS); opgrp++) begin : gen_operation_groups
    localparam int unsigned NUM_OPS = fpnew_pkg::num_operands(fpnew_pkg::opgroup_e'(opgrp));

    logic [NUM_FORMATS-1:0][NUM_OPS-1:0] input_boxed;
    fpnew_pkg::roundmode_e rnd_mode_in;
    fpnew_pkg::operation_e op_in;
    logic opmode_in;
    logic [NUM_OPERANDS-1:0][WIDTH-1:0] operands_input;
    TagType opgrp_tag_in;

    always_comb begin : select_opgrp_inputs
    unique case(tag_i.nl_op_sel)
      EXPS: begin
        if (nl_concatenate) begin
        concatenate_in_valid[opgrp] = (in_valid_i & (fpnew_pkg::get_opgroup(op_i) == fpnew_pkg::opgroup_e'(opgrp))) || (opgrp_out_valid[0] && (opgrp == 3));
        rnd_mode_in     = (opgrp == 3) ?  nl_rnd_conv    :  rnd_mode_i;
        op_in           = (opgrp == 3) ?  nl_op_conv     :  op_i;
        opmode_in       = (opgrp == 3) ?  nl_opmode_conv :  op_mod_i;
        operands_input  = (opgrp == 3) ?  operands_fconv :  operands_i;
        opgrp_tag_in    = (opgrp == 3) ?  fconv_tag      :  tag_i;
      end else begin
        concatenate_in_valid[opgrp] = in_valid_i & (fpnew_pkg::get_opgroup(op_i) == fpnew_pkg::opgroup_e'(opgrp));
        rnd_mode_in     = rnd_mode_i;
        op_in           = op_i;
        opmode_in       = op_mod_i;
        operands_input  = operands_i;
        opgrp_tag_in    = tag_i;
      end
      end
      COSHS: begin
        if (nl_concatenate) begin
        concatenate_in_valid[opgrp] = (in_valid_i & (fpnew_pkg::get_opgroup(op_i) == fpnew_pkg::opgroup_e'(opgrp))) || (opgrp_out_valid[0] && (opgrp == 3)) || (opgrp_out_valid[3] && (opgrp == 0));
        rnd_mode_in     = (opgrp == 3) ?  nl_rnd_conv    : ((opgrp_out_valid[3] && (opgrp == 0))) ? nl_rnd_add   : rnd_mode_i;
        op_in           = (opgrp == 3) ?  nl_op_conv     : ((opgrp_out_valid[3] && (opgrp == 0))) ? nl_op_add    : op_i;
        opmode_in       = (opgrp == 3) ?  nl_opmode_conv : ((opgrp_out_valid[3] && (opgrp == 0))) ? nl_opmode_add: op_mod_i;
        operands_input  = (opgrp == 3) ?  operands_fconv : ((opgrp_out_valid[3] && (opgrp == 0))) ? operands_add : operands_i;
        opgrp_tag_in    = (opgrp == 3) ?  fconv_tag      : ((opgrp_out_valid[3] && (opgrp == 0))) ? fadd_tag     : tag_i;
      end else begin
        concatenate_in_valid[opgrp] = in_valid_i & (fpnew_pkg::get_opgroup(op_i) == fpnew_pkg::opgroup_e'(opgrp));
        rnd_mode_in     = rnd_mode_i;
        op_in           = op_i;
        opmode_in       = op_mod_i;
        operands_input  = operands_i;
        opgrp_tag_in    = tag_i;
      end
      end
    endcase
      
    end
    // slice out input boxing
    always_comb begin : slice_inputs
      for (int unsigned fmt = 0; fmt < NUM_FORMATS; fmt++)
        input_boxed[fmt] = is_boxed[fmt][NUM_OPS-1:0];
    end

    fpnew_opgroup_block #(
      .OpGroup       ( fpnew_pkg::opgroup_e'(opgrp)    ),
      .Width         ( WIDTH                           ),
      .EnableVectors ( Features.EnableVectors          ),
      .DivSqrtSel    ( DivSqrtSel                      ),
      .FpFmtMask     ( Features.FpFmtMask              ),
      .IntFmtMask    ( Features.IntFmtMask             ),
      .FmtPipeRegs   ( Implementation.PipeRegs[opgrp]  ),
      .FmtUnitTypes  ( Implementation.UnitTypes[opgrp] ),
      .PipeConfig    ( Implementation.PipeConfig       ),
      .TagType       ( TagType                         ),
      .TrueSIMDClass ( TrueSIMDClass                   ),
      .CompressedVecCmpResult ( CompressedVecCmpResult ),
      .StochasticRndImplementation ( StochasticRndImplementation )
    ) i_opgroup_block (
      .clk_i,
      .rst_ni,
      .hart_id_i,
      .operands_i      ( operands_input[NUM_OPS-1:0] ),
      .is_boxed_i      ( input_boxed                 ),
      .rnd_mode_i      ( rnd_mode_in                 ),
      .op_i            ( op_in                       ),
      .op_mod_i        ( opmode_in                   ),
      .src_fmt_i,
      .dst_fmt_i,
      .int_fmt_i,
      .vectorial_op_i,
      .tag_i           ( opgrp_tag_in               ),
      .simd_mask_i     ( simd_mask                  ),
      .in_valid_i      ( concatenate_in_valid[opgrp]),
      .in_ready_o      ( opgrp_in_ready[opgrp]      ),
      .flush_i,
      .result_o        ( opgrp_outputs[opgrp].result ),
      .status_o        ( opgrp_outputs[opgrp].status ),
      .extension_bit_o ( opgrp_ext[opgrp]            ),
      .tag_o           ( opgrp_outputs[opgrp].tag    ),
      .out_valid_o     ( opgrp_out_valid[opgrp]      ),
      .out_ready_i     ( out_opgrp_ready[opgrp]      ),
      .busy_o          ( opgrp_busy[opgrp]           )
    );
    always_comb begin : mux_out_inp_concat
      if (nl_concatenate) begin
        unique case(tag_i.nl_op_sel)
          EXPS: concatenate_out_valid[opgrp] = opgrp_out_valid[opgrp] & (opgrp == 3);
          COSHS:begin 
            if (is_last_uop) concatenate_out_valid[opgrp] = opgrp_out_valid[opgrp] & (opgrp == 0) ; else concatenate_out_valid[opgrp] = '0;
          end
        endcase
      end else begin
        concatenate_out_valid[opgrp] =  opgrp_out_valid[opgrp];
      end
    end
  end

  // ------------------
  // Arbitrate Outputs
  // ------------------
  output_t arbiter_output;

  // Round-Robin arbiter to decide which result to use
  rr_arb_tree #(
    .NumIn     ( NUM_OPGROUPS ),
    .DataType  ( output_t     ),
    .AxiVldRdy ( 1'b1         )
  ) i_arbiter (
    .clk_i,
    .rst_ni,
    .flush_i,
    .rr_i   ( '0                    ),
    .req_i  ( concatenate_out_valid ),
    .gnt_o  ( opgrp_out_ready       ),
    .data_i ( opgrp_outputs         ),
    .gnt_i  ( out_ready_i           ),
    .req_o  ( out_valid_o           ),
    .data_o ( arbiter_output        ),
    .idx_o  ( /* unused */          )
  );


  // Unpack output
  assign result_o        = arbiter_output.result;
  assign status_o        = arbiter_output.status;
  assign tag_o           = arbiter_output.tag;

  assign busy_o = (| opgrp_busy);

endmodule
