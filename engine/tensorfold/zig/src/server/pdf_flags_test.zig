//! PDFium's render flags: annotations drawn, bytes left in BGRx order for the PNG encoder (upstream #2386, PR #2387).
const std = @import("std");
const pdf = @import("pdf.zig");

test "pdf render flags: annotations on, and not FPDF_REVERSE_BYTE_ORDER, which swapped red and blue" {
    try std.testing.expect(pdf.render_flags & 0x01 != 0); // FPDF_ANNOT
    try std.testing.expect(pdf.render_flags & 0x10 == 0); // FPDF_REVERSE_BYTE_ORDER: RGBx into a BGRx encoder
}
