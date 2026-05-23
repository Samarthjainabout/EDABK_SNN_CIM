module nvm_neuron_block(
  input               clk,
  input               rst,
  input signed [15:0] stimuli,
  input        [15:0] connection,
  input               picture_done,
  input               enable, 
  output       [15:0] spike_o
  );

  parameter NUM_OF_MACRO = 16;   
  
  // --- FINAL LIF TUNING ---
  // Threshold scaled up to match the 8-bit inflated weights (max 127)
  reg signed [15:0] THRESHOLD;

// Initialize it to our "Golden" 127 so it has a safe default
initial begin
    THRESHOLD = 16'sd127;
end
  // Very slow leak (voltage / 16) to clear out random noise over time
  parameter [3:0] LEAK_SHIFT = 4'd4; 

  reg signed [15:0] potential [NUM_OF_MACRO-1:0]; 

  genvar i;
  generate
    for (i = 0; i < NUM_OF_MACRO; i=i+1) begin
      always @(posedge clk or posedge rst) begin
        if (rst) 
            potential[i] <= 16'b0;
        else if (picture_done) 
            potential[i] <= 16'b0;
        else if (spike_o[i]) // <--- NEW: Reset after firing
            potential[i] <= 16'b0; 
        else if (enable & connection[i]) 
            // Simple integration with a slow leak. No artificial gain needed.
            potential[i] <= potential[i] - (potential[i] >>> LEAK_SHIFT) + stimuli;
        else 
            // Leak only (even if no input spike, the neuron should still leak)
            potential[i] <= potential[i] - (potential[i] >>> LEAK_SHIFT);
      end

      assign spike_o[i] = ($signed(potential[i]) >= THRESHOLD);
    end
  endgenerate

endmodule