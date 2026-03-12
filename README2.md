# CONV2D ACCELERATOR: FULL SoC INTEGRATION ON KV260 — DEEP DIVE
## Line Buffers, Sliding Windows, AXI Interfaces, DMA, Vivado Block Design & Clocking

> **Context:** This file covers the SoC-level integration details of the FPGA Conv2D Accelerator project that are NOT already in `Day5_Project_Deep_Dive.md`. Day5 covers the CNN basics, Vitis HLS kernel overview, and general Q&A. This file goes deeper into the hardware architecture and Vivado integration.

---

# TABLE OF CONTENTS
1. [Conv2D Micro-Architecture: Line Buffers & Sliding Windows](#1-conv2d-micro-architecture)
2. [AXI4-Stream Interface Design](#2-axi4-stream-interface-design)
3. [AXI-Lite Control Interface](#3-axi-lite-control-interface)
4. [AXI DMA for DDR-Based Streaming](#4-axi-dma-for-ddr-based-streaming)
5. [Complete SoC Block Design in Vivado](#5-complete-soc-block-design-in-vivado)
6. [Clocking, Resets & Address Mapping](#6-clocking-resets--address-mapping)
7. [PS–PL Data & Control Paths](#7-pspl-data--control-paths)
8. [Vitis HLS Pragmas Deep Dive](#8-vitis-hls-pragmas-deep-dive)
9. [Resource Utilization on KV260](#9-resource-utilization-on-kv260)
10. [End-to-End Data Flow Walkthrough](#10-end-to-end-data-flow-walkthrough)
11. [Interview Q&A — SoC Integration Focus](#11-interview-qa--soc-integration-focus)

---

# 1. CONV2D MICRO-ARCHITECTURE: LINE BUFFERS & SLIDING WINDOWS

## The Problem Conv2D Solves

A 2D convolution slides a K×K kernel across an H×W feature map. The naïve approach re-fetches overlapping pixels from external memory, wasting bandwidth. Line buffers and a sliding window cache exactly the rows and columns needed on-chip.

## Line Buffer Concept

```
For a 3×3 kernel operating on a W-wide image:

External Memory (DDR):
  Row 0:  p00  p01  p02  p03  p04  ...  p0(W-1)
  Row 1:  p10  p11  p12  p13  p14  ...  p1(W-1)
  Row 2:  p20  p21  p22  p23  p24  ...  p2(W-1)
  Row 3:  p30  p31  p32  p33  p34  ...  p3(W-1)
  ...

Line Buffer (on-chip BRAM/LUTRAM):
  ┌────────────────────────────────────┐
  │ LB[0]: | p00 | p01 | p02 | p03 | ... | p0(W-1) |  ← oldest row
  │ LB[1]: | p10 | p11 | p12 | p13 | ... | p1(W-1) |  ← middle row
  └────────────────────────────────────┘
  Incoming pixel stream feeds the CURRENT row directly.

Only K-1 = 2 full rows are buffered.
The K-th row comes from the live input stream.
Total on-chip storage: (K-1) × W pixels.
```

## Sliding Window Register Array

```
The sliding window is a K×K register array that holds the current
convolution neighbourhood.

For 3×3 kernel at column position c:

  ┌──────────┬──────────┬──────────┐
  │ LB[0][c] │ LB[0][c+1] │ LB[0][c+2] │   ← from line buffer 0
  ├──────────┼──────────┼──────────┤
  │ LB[1][c] │ LB[1][c+1] │ LB[1][c+2] │   ← from line buffer 1
  ├──────────┼──────────┼──────────┤
  │ stream[c]│ stream[c+1]│ stream[c+2]│   ← from live input
  └──────────┴──────────┴──────────┘

Each clock cycle:
  1. Shift window columns left by 1
  2. Insert new pixels from line buffers + stream into rightmost column
  3. Perform K×K MAC (multiply-accumulate) over the window
  4. Emit one output pixel

This achieves an Initiation Interval (II) of 1 → one output per clock.
```

## HLS Implementation of Line Buffer + Sliding Window

```cpp
#include "ap_int.h"
#include "hls_stream.h"

#define K       3       // Kernel size
#define MAX_W   1920    // Maximum image width
#define MAX_H   1080

typedef ap_fixed<16,8> data_t;
typedef ap_fixed<16,8> weight_t;
typedef ap_fixed<32,16> acc_t;

void conv2d_linebuffer(
    hls::stream<data_t> &in_stream,
    hls::stream<data_t> &out_stream,
    weight_t kernel[K][K],
    int width, int height
) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=width   bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=height  bundle=ctrl
    #pragma HLS INTERFACE s_axilite port=return  bundle=ctrl

    // Line buffers — store (K-1) rows
    data_t line_buf[K-1][MAX_W];
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    // dim=1 partition: each row is a separate BRAM → parallel access

    // Sliding window — K×K register array
    data_t window[K][K];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0
    // complete dim=0: all elements become registers → fully parallel

    ROW_LOOP: for (int r = 0; r < height; r++) {
        COL_LOOP: for (int c = 0; c < width; c++) {
            #pragma HLS PIPELINE II=1

            // 1. Read new pixel from stream
            data_t new_pixel = in_stream.read();

            // 2. Shift window columns left
            for (int kr = 0; kr < K; kr++) {
                for (int kc = 0; kc < K-1; kc++) {
                    window[kr][kc] = window[kr][kc+1];
                }
            }

            // 3. Insert new column from line buffers + new pixel
            for (int kr = 0; kr < K-1; kr++) {
                window[kr][K-1] = line_buf[kr][c];
            }
            window[K-1][K-1] = new_pixel;

            // 4. Update line buffers (shift rows up)
            for (int kr = 0; kr < K-2; kr++) {
                line_buf[kr][c] = line_buf[kr+1][c];
            }
            line_buf[K-2][c] = new_pixel;

            // 5. Compute convolution (MAC)
            acc_t acc = 0;
            for (int kr = 0; kr < K; kr++) {
                for (int kc = 0; kc < K; kc++) {
                    acc += window[kr][kc] * kernel[kr][kc];
                }
            }

            // 6. Write output (valid only when window is fully populated)
            if (r >= K-1 && c >= K-1) {
                out_stream.write((data_t)acc);
            }
        }
    }
}
```

### Why This Architecture Is Efficient

| Metric | Naïve (DDR re-fetch) | Line Buffer + Sliding Window |
|--------|----------------------|------------------------------|
| DDR reads per pixel | K×K = 9 (for 3×3) | **1** (each pixel read once) |
| On-chip storage | 0 | (K-1)×W + K² registers |
| Throughput | Limited by DDR BW | **1 pixel/cycle** (II=1) |
| Latency (first output) | K×K clocks | (K-1)×W + K clocks (pipeline fill) |

---

# 2. AXI4-STREAM INTERFACE DESIGN

## AXI4-Stream Signal Breakdown

```
 PS (DMA) ──── AXI4-Stream ────→ Conv2D IP (PL)

Signal     Width     Direction   Purpose
───────    ─────     ─────────   ──────────────────────────────
TDATA      32-bit    M → S       Pixel data (e.g., 16-bit fixed-point in lower half)
TVALID     1-bit     M → S       Master asserts when TDATA is valid
TREADY     1-bit     S → M       Slave asserts when it can accept data
TLAST      1-bit     M → S       Marks last transfer in a packet/frame
TKEEP      4-bit     M → S       Byte qualifiers (which bytes of TDATA are valid)
TDEST      -         M → S       Routing info (unused in simple designs)
TID        -         M → S       Stream identifier (unused in simple designs)

Handshake: Transfer occurs when TVALID=1 AND TREADY=1 on rising clock edge.
```

## Back-Pressure Mechanism

```
Clock:    __|‾‾|__|‾‾|__|‾‾|__|‾‾|__|‾‾|__|‾‾|__|‾‾|
TVALID:   ___/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\___
TREADY:   ___/‾‾‾‾‾‾\______/‾‾‾‾‾‾‾‾‾‾‾‾\___
TDATA:       D0  D1   stall  D2  D3  D4
                     ↑
              Slave de-asserts TREADY → DMA stalls
              No data is lost; master holds TDATA stable

This is how the Conv2D IP can throttle the DMA if it falls behind:
  - Conv2D is slow → de-asserts TREADY → DMA pauses
  - Conv2D catches up → re-asserts TREADY → DMA resumes
```

## Vitis HLS → AXI4-Stream Generation

```cpp
// HLS pragma automatically generates AXI4-Stream ports
#pragma HLS INTERFACE axis port=in_stream
// This generates: in_stream_TDATA, in_stream_TVALID, in_stream_TREADY, in_stream_TLAST

// You can also bundle TLAST:
#pragma HLS INTERFACE axis port=in_stream register_mode=both
```

---

# 3. AXI-LITE CONTROL INTERFACE

## What AXI-Lite Does in This Design

AXI-Lite is a lightweight memory-mapped interface (no burst, single-beat). The PS (ARM CPU) uses it to:
1. Write configuration registers (image width, height, kernel weights)
2. Start the accelerator (ap_start)
3. Poll for completion (ap_done, ap_idle)

## AXI-Lite Register Map (Auto-Generated by HLS)

```
Offset   Register         Access   Description
──────   ──────────       ──────   ─────────────────────────────
0x00     AP_CTRL          R/W      Bit 0: ap_start, Bit 1: ap_done, Bit 2: ap_idle
0x04     GIE              R/W      Global Interrupt Enable
0x08     IER              R/W      IP Interrupt Enable Register
0x0C     ISR              R/W      IP Interrupt Status Register
0x10     width            W        Image width (32-bit)
0x18     height           W        Image height (32-bit)
0x20     kernel_0_0       W        Kernel weight [0][0]
0x28     kernel_0_1       W        Kernel weight [0][1]
...      ...              ...      (9 weights for 3×3 kernel)
```

## PS-Side C Code to Program AXI-Lite

```c
#include "xconv2d_linebuffer.h"  // Auto-generated driver from HLS

XConv2d_linebuffer conv_ip;

// Initialize IP
XConv2d_linebuffer_Initialize(&conv_ip, XPAR_CONV2D_LINEBUFFER_0_DEVICE_ID);

// Set parameters via AXI-Lite
XConv2d_linebuffer_Set_width(&conv_ip, 1920);
XConv2d_linebuffer_Set_height(&conv_ip, 1080);

// Load kernel weights
XConv2d_linebuffer_Write_kernel_Words(&conv_ip, 0, (u32 *)kernel_data, 9);

// Start IP
XConv2d_linebuffer_Start(&conv_ip);

// Poll for completion
while (!XConv2d_linebuffer_IsDone(&conv_ip));
printf("Conv2D complete!\n");
```

---

# 4. AXI DMA FOR DDR-BASED STREAMING

## AXI DMA Architecture in This Design

```
                      AXI4 Memory-Mapped                AXI4-Stream
  ┌────────┐         (Full)                             (to/from PL IP)
  │  DDR4  │ ←────→ ┌──────────────┐  MM2S stream  ──→ ┌──────────────┐
  │ Memory │         │  AXI DMA     │                    │  Conv2D      │
  │        │ ←────→ │  Controller  │  S2MM stream  ←── │  Accelerator │
  └────────┘         └──────────────┘                    └──────────────┘
                         ↑                                      ↑
                    AXI-Lite                               AXI-Lite
                   (PS control)                          (PS control)
                         ↑                                      ↑
                    ┌────┴────────────────────────────────────────┘
                    │     PS (ARM Cortex-A53)
                    │     Running Linux / Bare-metal App
                    └────────────────────────────────────────────

DMA Channels:
  MM2S (Memory-Mapped to Stream): Reads image from DDR → pushes to Conv2D input
  S2MM (Stream to Memory-Mapped): Receives Conv2D output → writes to DDR
```

## DMA Transfer Setup (PS Side)

```c
#include "xaxidma.h"

XAxiDma dma;
XAxiDma_Config *dma_cfg;

// Initialize DMA
dma_cfg = XAxiDma_LookupConfig(XPAR_AXIDMA_0_DEVICE_ID);
XAxiDma_CfgInitialize(&dma, dma_cfg);

// Disable interrupts for simple polling mode
XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);

// Flush cache (CRITICAL: ensure DDR data is coherent)
Xil_DCacheFlushRange((UINTPTR)src_buffer, image_size);

// Start MM2S transfer: DDR → PL (send image to Conv2D)
XAxiDma_SimpleTransfer(&dma, (UINTPTR)src_buffer, image_size, XAXIDMA_DMA_TO_DEVICE);

// Start S2MM transfer: PL → DDR (receive Conv2D output)
XAxiDma_SimpleTransfer(&dma, (UINTPTR)dst_buffer, output_size, XAXIDMA_DEVICE_TO_DMA);

// Wait for both transfers to complete
while (XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE));
while (XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA));

// Invalidate cache before reading output
Xil_DCacheInvalidateRange((UINTPTR)dst_buffer, output_size);
```

## Why DMA Matters for Performance

```
Without DMA (CPU Copy):
  CPU reads each pixel from DDR → writes to PL FIFO → CPU is 100% busy
  Throughput: ~50-100 MB/s (limited by CPU load/store)

With DMA:
  DMA engine handles the transfer autonomously
  CPU only sets up source, destination, size → then CPU is FREE
  Throughput: ~1-4 GB/s (limited by DDR bandwidth)
  CPU can prepare next frame or run other tasks concurrently
```

---

# 5. COMPLETE SoC BLOCK DESIGN IN VIVADO

## Vivado IP Integrator Block Design

```
┌──────────────────────────────────────────────────────────────────────┐
│                     VIVADO BLOCK DESIGN                              │
│                                                                      │
│  ┌─────────────────────────┐                                         │
│  │   Zynq UltraScale+     │                                         │
│  │    PS (Processing       │                                         │
│  │     System)             │                                         │
│  │                         │                                         │
│  │  M_AXI_HPM0_FPD ───────┼──→ AXI Interconnect ──→ DMA (AXI-Lite) │
│  │                         │                     ──→ Conv2D (AXI-Lite)│
│  │  S_AXI_HP0_FPD  ←──────┼──← AXI Interconnect ←── DMA (AXI-MM)   │
│  │                         │                                         │
│  │  pl_clk0 (100 MHz) ────┼──→ Clocking Wizard ──→ IP clocks       │
│  │  pl_resetn0 ───────────┼──→ Proc Sys Reset ──→ IP resets        │
│  └─────────────────────────┘                                         │
│                                                                      │
│  ┌──────────────┐   AXI4-Stream    ┌──────────────┐                  │
│  │   AXI DMA    │ ───MM2S_AXIS───→ │  Conv2D      │                  │
│  │              │ ←──S2MM_AXIS──── │  Linebuffer  │                  │
│  │              │                   │  HLS IP      │                  │
│  └──────────────┘                   └──────────────┘                  │
│                                                                      │
│  ┌──────────────┐   ┌──────────────────┐                             │
│  │  Clocking    │   │  Processor       │                             │
│  │  Wizard      │   │  System Reset    │                             │
│  │ (optional)   │   │  (proc_sys_reset)│                             │
│  └──────────────┘   └──────────────────┘                             │
└──────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Vivado Integration Flow

```
1. Export HLS IP
   Vitis HLS → Solution → Export RTL (IP Catalog format .zip)
   Copy to IP repository path

2. Create Block Design in Vivado
   File → Create Block Design → "conv2d_system"

3. Add Zynq UltraScale+ PS
   Add IP → "Zynq UltraScale+ MPSoC"
   Run Block Automation → configures DDR, I/O, clocks

4. Add HLS IP (Conv2D)
   Settings → IP Repository → add HLS export path
   Add IP → "conv2d_linebuffer"

5. Add AXI DMA
   Add IP → "AXI Direct Memory Access"
   Configure: Enable MM2S and S2MM channels
   Set width: 32-bit data, 32-bit address
   Buffer Length Register Width: 26 bits (max 64MB transfer)

6. Run Connection Automation
   Vivado auto-connects AXI-Lite control ports through AXI Interconnect
   Vivado auto-connects DMA memory-mapped port to PS HP port

7. Manual Connections
   DMA MM2S_AXIS → Conv2D in_stream
   Conv2D out_stream → DMA S2MM_AXIS
   Connect interrupt lines: DMA mm2s_introut, s2mm_introut → PS IRQ

8. Add Clock & Reset IPs
   Proc System Reset block for synchronized resets
   (Optional) Clocking Wizard if IP needs different clock frequency

9. Validate Design
   Tools → Validate Design (checks all connections, address conflicts)

10. Generate Output Products
    Right-click block design → Generate Output Products

11. Create HDL Wrapper
    Right-click block design → Create HDL Wrapper

12. Synthesize → Implement → Generate Bitstream

13. Export Hardware (.xsa)
    File → Export Hardware → Include Bitstream
    This .xsa file is used in Vitis IDE for PS application development
```

---

# 6. CLOCKING, RESETS & ADDRESS MAPPING

## Clock Architecture

```
PS pl_clk0 (100 MHz default)
    │
    ├──→ AXI DMA (aclk)
    ├──→ AXI Interconnect (aclk)
    ├──→ Conv2D IP (ap_clk)
    └──→ Proc System Reset (slowest_sync_clk)

All AXI interfaces MUST share the same clock domain
(or use AXI Clock Converter IP for cross-domain).

For higher-performance Conv2D:
  PS pl_clk0 → Clocking Wizard → 200 MHz output → Conv2D ap_clk
  BUT: then you need AXI Clock Converter between DMA and Conv2D
  Trade-off: higher throughput vs. more resources & complexity
```

## Reset Architecture

```
PS pl_resetn0 (active-low)
    │
    └──→ Processor System Reset
            │
            ├──→ peripheral_aresetn[0] ──→ AXI DMA aresetn
            ├──→ peripheral_aresetn[0] ──→ AXI Interconnect aresetn
            ├──→ peripheral_aresetn[0] ──→ Conv2D ap_rst_n
            └──→ interconnect_aresetn[0] ──→ AXI Interconnect ARESETN

The Proc System Reset IP:
  - Synchronizes the async PS reset to the PL clock domain
  - Generates separate reset signals for peripherals vs. interconnect
  - Ensures minimum 16-cycle reset pulse width
  - Holds reset until PL clocks are stable (dcm_locked input)
```

## Address Mapping

```
Vivado Address Editor:

Master: PS M_AXI_HPM0_FPD
  ├── AXI DMA (S_AXI_LITE)
  │     Base: 0xA000_0000    Size: 64K    High: 0xA000_FFFF
  ├── Conv2D IP (s_axi_ctrl)
  │     Base: 0xA001_0000    Size: 64K    High: 0xA001_FFFF
  └── (Auto-assigned by Vivado, can be manually overridden)

Master: AXI DMA (M_AXI_MM2S, M_AXI_S2MM)
  └── PS S_AXI_HP0_FPD
        Base: 0x0000_0000    Size: 2G     High: 0x7FFF_FFFF
        (Full DDR address space accessible by DMA)

These addresses are used in the PS driver code:
  #define XPAR_AXIDMA_0_BASEADDR      0xA0000000
  #define XPAR_CONV2D_0_S_AXI_CTRL    0xA0010000
```

---

# 7. PS–PL DATA & CONTROL PATHS

## Complete Data Path

```
CONTROL PATH (AXI-Lite, low bandwidth):
  PS CPU ──→ AXI Interconnect ──→ DMA Control Registers (setup transfer)
  PS CPU ──→ AXI Interconnect ──→ Conv2D Control Registers (set width/height/start)

DATA PATH (AXI4-Full + AXI4-Stream, high bandwidth):
  DDR ←──AXI4-Full──→ DMA ←──AXI4-Stream──→ Conv2D IP

  Read path:  DDR →(AXI4-Full)→ DMA →(MM2S AXI-Stream)→ Conv2D
  Write path: Conv2D →(S2MM AXI-Stream)→ DMA →(AXI4-Full)→ DDR
```

## PS Application Execution Order

```c
// 1. Initialize hardware
init_platform();

// 2. Configure Conv2D IP via AXI-Lite
XConv2d_Set_width(&conv, 640);
XConv2d_Set_height(&conv, 480);
load_kernel_weights(&conv, sobel_3x3);

// 3. Prepare input data in DDR
load_image_to_ddr(src_buffer, "input.raw");

// 4. Flush data cache (ensure DDR coherency)
Xil_DCacheFlushRange((UINTPTR)src_buffer, img_size);

// 5. Start Conv2D IP (it waits for stream data)
XConv2d_Start(&conv);

// 6. Start DMA transfers (data starts flowing)
XAxiDma_SimpleTransfer(&dma, (UINTPTR)src_buffer, img_size, XAXIDMA_DMA_TO_DEVICE);
XAxiDma_SimpleTransfer(&dma, (UINTPTR)dst_buffer, out_size, XAXIDMA_DEVICE_TO_DMA);

// 7. Wait for completion
while (XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE));
while (XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA));

// 8. Invalidate cache, read results
Xil_DCacheInvalidateRange((UINTPTR)dst_buffer, out_size);
verify_output(dst_buffer);
```

---

# 8. VITIS HLS PRAGMAS DEEP DIVE

## Pragmas Used in This Design

| Pragma | Purpose | Effect on Hardware |
|--------|---------|-------------------|
| `#pragma HLS INTERFACE axis` | AXI4-Stream port | Generates TDATA/TVALID/TREADY/TLAST signals |
| `#pragma HLS INTERFACE s_axilite` | AXI-Lite port | Memory-mapped register for PS control |
| `#pragma HLS PIPELINE II=1` | Pipeline inner loop | One new input per clock cycle |
| `#pragma HLS ARRAY_PARTITION complete dim=1` | Split array into separate BRAMs | Parallel row access for line buffers |
| `#pragma HLS ARRAY_PARTITION complete dim=0` | Split array into registers | Fully parallel access (window registers) |
| `#pragma HLS DATAFLOW` | Task-level pipelining | Multiple functions execute concurrently |
| `#pragma HLS UNROLL` | Unroll loop iterations | Replicate hardware for parallel execution |
| `#pragma HLS BIND_STORAGE` | Specify memory type | Force BRAM, LUTRAM, or registers |

## PIPELINE II=1 Explained

```
Without PIPELINE:
  Cycle:   1    2    3    4    5    6    7    8    9
  Pixel0: [RD] [MAC] [WR]
  Pixel1:                [RD] [MAC] [WR]
  Pixel2:                               [RD] [MAC] [WR]
  Throughput: 1 pixel every 3 cycles

With PIPELINE II=1:
  Cycle:   1    2    3    4    5    6
  Pixel0: [RD] [MAC] [WR]
  Pixel1:      [RD]  [MAC] [WR]
  Pixel2:            [RD]  [MAC] [WR]
  Pixel3:                  [RD]  [MAC] [WR]
  Throughput: 1 pixel every 1 cycle (after pipeline fill)
  
  At 200 MHz clock: 200 million pixels/second!
```

## DATAFLOW — Task-Level Pipelining

```cpp
void top_function(hls::stream<data_t> &in, hls::stream<data_t> &out) {
    #pragma HLS DATAFLOW
    
    hls::stream<data_t> tmp1, tmp2;
    
    read_input(in, tmp1);       // Stage 1
    conv2d_compute(tmp1, tmp2); // Stage 2
    write_output(tmp2, out);    // Stage 3
}

// Without DATAFLOW: Stage1 finishes → Stage2 starts → Stage3 starts (sequential)
// With DATAFLOW:    All 3 stages run concurrently, connected by FIFOs
//   Frame N:   [read_input] → [conv2d_compute] → [write_output]
//   Frame N+1:   [read_input] → [conv2d_compute] → ...
//   Throughput improvement: up to 3× for multi-stage pipelines
```

---

# 9. RESOURCE UTILIZATION ON KV260

## Typical Resource Usage (3×3 Conv2D, 1920-wide image)

```
+-------------------+--------+-----------+-------------+
| Resource          | Used   | Available | Utilization |
+-------------------+--------+-----------+-------------+
| LUT               | ~2,500 | 117,120   | ~2.1%       |
| LUTRAM            | ~200   | 57,600    | ~0.3%       |
| FF (Flip-Flops)   | ~3,000 | 234,240   | ~1.3%       |
| BRAM (36Kb)       | ~8     | 144       | ~5.6%       |
| DSP48E2           | ~9     | 1,248     | ~0.7%       |
+-------------------+--------+-----------+-------------+

Notes:
  - 9 DSP slices → one per kernel weight (3×3 = 9 MACs in parallel)
  - 8 BRAMs → line buffers (2 rows × 1920 pixels × 16-bit ≈ 7.5 KB per row)
  - Very low utilization → room for multi-channel, multi-layer expansion
```

## Scaling to Multiple Output Channels

```
Single channel:  9 DSPs,  ~8 BRAMs
16 channels:     144 DSPs, ~128 BRAMs (16 parallel Conv2D units)
64 channels:     576 DSPs, ~512 BRAMs (approaching KV260 limits)

Strategy for large networks:
  - Time-multiplex channels (reuse same hardware, process one channel at a time)
  - Tile spatial dimensions (process image in sub-tiles)
  - Layer fusion (pipeline multiple layers without DDR round-trip)
```

---

# 10. END-TO-END DATA FLOW WALKTHROUGH

```
1. PS loads input image (640×480×1 grayscale) into DDR at address 0x1000_0000
2. PS writes Conv2D config via AXI-Lite:
     width=640, height=480, kernel weights={edge detection filter}
3. PS writes AP_CTRL.ap_start = 1 → Conv2D IP enters READY state, waiting for stream
4. PS programs DMA:
     MM2S: source=0x1000_0000, length=640×480×2=614,400 bytes
     S2MM: dest=0x2000_0000, length=638×478×2=610,228 bytes (output smaller by K-1)
5. DMA starts MM2S: reads burst from DDR → pushes pixels on AXI4-Stream
6. Conv2D receives pixels:
     - First (K-1)×W = 2×640 = 1280 pixels fill line buffers (no output yet)
     - From pixel 1281 onward: one output per clock cycle
7. Conv2D pushes output pixels on S2MM AXI4-Stream
8. DMA S2MM channel writes output pixels to DDR at 0x2000_0000
9. DMA completion → interrupt or polling flag
10. PS invalidates cache → reads result from DDR → displays/verifies
```

---

# 11. INTERVIEW Q&A — SoC INTEGRATION FOCUS

**Q: Why did you use line buffers instead of storing the entire image on-chip?**
A: A 1920×1080 image at 16-bit per pixel requires ~4 MB. The KV260's BRAM is only ~5 MB total. Line buffers store only K-1 rows (~7.5 KB for 3×3 on 1920-wide), reducing BRAM usage by >500× while maintaining II=1 throughput.

**Q: How does the AXI DMA know when to stop transferring?**
A: The PS programs the DMA with a specific byte count. The DMA counts transferred bytes and stops when the count is reached. Additionally, TLAST on the stream marks packet boundaries. The S2MM channel can also be configured to stop on TLAST.

**Q: What happens if the Conv2D IP is slower than the DMA?**
A: The AXI4-Stream back-pressure mechanism handles this. The Conv2D IP de-asserts TREADY, causing the DMA to pause. No data is lost — the DMA simply waits. This is the beauty of the TVALID/TREADY handshake.

**Q: Why do you need to flush/invalidate the data cache?**
A: The ARM CPU has L1/L2 caches. When the CPU writes image data to DDR, it may sit in the cache, not in actual DDR. The DMA reads from DDR directly (bypassing cache). So we flush the cache to push data to DDR before DMA reads. Similarly, after DMA writes output to DDR, the CPU cache may have stale data, so we invalidate it before CPU reads.

**Q: What is the difference between AXI4-Full and AXI4-Stream?**
A: AXI4-Full is memory-mapped with addresses — used for random access to DDR. It supports burst transfers (up to 256 beats). AXI4-Stream has no addresses — it's a unidirectional point-to-point data pipe optimized for continuous streaming. In our design, DMA bridges between them: AXI4-Full on the DDR side, AXI4-Stream on the accelerator side.

**Q: How did you verify the SoC design?**
A: Multiple levels: (1) HLS C/RTL co-simulation to verify the Conv2D kernel alone, (2) Vivado block design validation to check connectivity, (3) Vivado timing simulation post-implementation, (4) On-board testing with known input images and comparing output against MATLAB golden reference.

**Q: What is the Processor System Reset IP and why is it needed?**
A: The PS generates an asynchronous reset. The PL runs on a different clock (pl_clk0). The Proc System Reset IP synchronizes the reset to the PL clock domain, ensures a minimum reset pulse width, and waits until the PL clock is stable before releasing reset. Without it, PL IPs could come out of reset in an undefined state.

**Q: Could you scale this design to handle RGB (3-channel) images?**
A: Yes, three approaches: (1) Process channels sequentially — simple but 3× slower. (2) Instantiate 3 parallel Conv2D units — 3× resources but same throughput. (3) Interleave channels in a single wider datapath with TDATA=96-bit (3×32) — best balance of throughput and resources.

---

*This file complements `Day5_Project_Deep_Dive.md` (which covers CNN basics and general HLS) with the deep SoC integration details.*
