# Downloads (per subject) the planned MESA subset (results/mesa/audit/subset_plan.csv) into data/mesa/.
# Token from ENV only. gem verifies md5 per file (re-downloads once on mismatch).
# Disk guard: aborts if root fs >= 85% used. Re-runnable (md5 skip). Pipe output through a token-scrubbing sed.
module Nsrr; end
require "nsrr/version"
require "nsrr/models/all"
require "csv"
tok = ENV["NSRR_TOKEN"] or abort("NSRR_TOKEN unset")
ids = CSV.read("results/mesa/audit/subset_plan.csv", headers: true).map { |r| r["mesaid"] }
def retry_call(n = 6) ; n.times { |i| r = yield; return r if r; sleep 15 * (i + 1) }; nil; end
def disk_pct; `df --output=pcent / | tail -1`.to_i; end
ds = retry_call { Nsrr::Models::Dataset.find("mesa", tok) } or abort("dataset lookup failed")
targets = [["polysomnography/edfs", "mesa-sleep-%s.edf"],
           ["polysomnography/annotations-events-nsrr", "mesa-sleep-%s-nsrr.xml"],
           ["polysomnography/annotations-events-profusion", "mesa-sleep-%s-profusion.xml"]]
Dir.chdir("data")
failed = []
consec = 0
# harmonized/dataset CSVs first (small)
["datasets/mesa-sleep-dataset-0.8.0.csv", "datasets/mesa-sleep-harmonized-dataset-0.8.0.csv",
 "datasets/mesa-data-dictionary-0.8.0-variables.csv"].each do |fp|
  folder = File.dirname(fp)
  f = retry_call { ds.files(folder).find { |x| x.full_path == fp } }
  FileUtils.mkdir_p("mesa/#{folder}")
  r = f && f.download("md5", "mesa/#{folder}", tok)
  failed << fp if r.nil? || r == "fail"
end
# PER SUBJECT: EDF + NSRR event XML + Profusion XML together, so a complete subject lands every ~10 min.
listings = {}
targets.each do |folder, pat|
  listing = retry_call { l = ds.files(folder); l.empty? ? nil : l } or abort("listing #{folder} failed")
  listings[folder] = listing.select(&:is_file).map { |f| [f.file_name, f] }.to_h
  FileUtils.mkdir_p("mesa/#{folder}")
end
ids.each do |id|
  targets.each do |folder, pat|
    abort("DISK GUARD: #{disk_pct}% used, stopping") if disk_pct >= 85
    fn = pat % id
    lf = listings[folder][fn]
    (failed << "#{fn}(not listed)"; next) unless lf
    next if File.exist?("mesa/#{folder}/#{fn}") && File.size("mesa/#{folder}/#{fn}") == lf.file_size
    ok = false
    2.times do |i|
      r = lf.download("md5", "mesa/#{folder}", tok)
      (ok = true; break) unless r == "fail"
      sleep 20 * (i + 1)
    end
    consec = ok ? 0 : consec + 1
    abort("SERVER DOWN: #{consec} consecutive failures, exiting for wrapper backoff") if consec >= 4
    failed << fn unless ok
    $stdout.flush
  end
  puts "SUBJECT_COMPLETE #{id}" if targets.all? { |f, pat| File.exist?("mesa/#{f}/#{pat % id}") }
  $stdout.flush
end
puts "DONE failed=#{failed.size} #{failed.first(20).inspect}"
