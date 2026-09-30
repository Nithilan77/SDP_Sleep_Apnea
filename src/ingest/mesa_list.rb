# Lists NSRR mesa tree (no downloads). Token read only from ENV. Usage: ruby mesa_list.rb [path]
module Nsrr; end
require "nsrr/models/all"
tok = ENV["NSRR_TOKEN"] or abort("NSRR_TOKEN unset")
ds = Nsrr::Models::Dataset.find("mesa", tok) or abort("dataset lookup failed (network/token)")
fs = ds.files(ARGV[0])
fs.first(ARGV[1] ? ARGV[1].to_i : 60).each { |f| puts [f.is_file ? "F" : "D", f.full_path, f.file_size, f.file_name].join("\t") }
puts "TOTAL entries: #{fs.size}"
