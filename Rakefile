require 'rake/extensiontask'

Rake::ExtensionTask.new('daimond_rust') do |ext|
  ext.lib_dir = 'lib/daimond/rust'
  ext.source_pattern = 'ext/daimond_rust/src/lib.rs'
end

task default: :compile