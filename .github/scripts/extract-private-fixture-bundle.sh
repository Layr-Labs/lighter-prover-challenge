#!/bin/bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: extract-private-fixture-bundle.sh BUNDLE_PATH OUTPUT_DIR EXPECTED_COUNT" >&2
  exit 2
fi

bundle="$1"
output="$2"
expected_count="$3"
output_owned=0

fail() {
  echo "private fixture bundle: validation failed" >&2
  exit 1
}

cleanup() {
  local status=$?
  trap - EXIT HUP INT TERM
  if [[ "${output_owned}" == 1 ]]; then
    /bin/rm -rf -- "${output}" >/dev/null 2>&1 || true
  fi
  exit "${status}"
}

interrupted() {
  exit 1
}

[[ "${expected_count}" =~ ^[1-9][0-9]*$ ]] || fail
[[ -n "${bundle}" && -n "${output}" ]] || fail
[[ ! -e "${output}" && ! -L "${output}" ]] || fail

trap cleanup EXIT
trap interrupted HUP INT TERM

if ! /usr/bin/perl -MArchive::Tar -MFcntl=:mode -e '
  use strict;
  use warnings;

  my ($bundle, $expected_count) = @ARGV;
  my @metadata = lstat($bundle);
  exit 1 unless @metadata;
  exit 1 unless S_ISREG($metadata[2]);
  exit 1 unless ($metadata[2] & 07777) == 0600;

  my $archive = Archive::Tar->new;
  exit 1 unless $archive->read($bundle, Archive::Tar::COMPRESS_GZIP());
  my @entries = $archive->get_files;
  exit 1 unless @entries == $expected_count;

  my %seen;
  for my $entry (@entries) {
    my $name = $entry->full_path;
    exit 1 unless $entry->is_file;
    exit 1 unless defined($name);
    exit 1 unless $name =~ /\A[A-Za-z0-9][A-Za-z0-9_-]*[.]json\z/;
    exit 1 if $name =~ m{/};
    exit 1 if $name =~ /[[:cntrl:]]/;
    exit 1 if $seen{$name}++;
  }
' "${bundle}" "${expected_count}" >/dev/null 2>&1; then
  fail
fi

umask 077
if ! /bin/mkdir -m 0700 -- "${output}" >/dev/null 2>&1; then
  fail
fi
output_owned=1

if ! /usr/bin/perl -MArchive::Tar -MFcntl=:DEFAULT,:mode -e '
  use strict;
  use warnings;

  my ($bundle, $output, $expected_count) = @ARGV;
  my @bundle_metadata = lstat($bundle);
  exit 1 unless @bundle_metadata;
  exit 1 unless S_ISREG($bundle_metadata[2]);
  exit 1 unless ($bundle_metadata[2] & 07777) == 0600;

  my @output_metadata = lstat($output);
  exit 1 unless @output_metadata;
  exit 1 unless S_ISDIR($output_metadata[2]);
  exit 1 unless ($output_metadata[2] & 07777) == 0700;

  my $archive = Archive::Tar->new;
  exit 1 unless $archive->read($bundle, Archive::Tar::COMPRESS_GZIP());
  my @entries = $archive->get_files;
  exit 1 unless @entries == $expected_count;

  my %seen;
  for my $entry (@entries) {
    my $name = $entry->full_path;
    exit 1 unless $entry->is_file;
    exit 1 unless defined($name);
    exit 1 unless $name =~ /\A[A-Za-z0-9][A-Za-z0-9_-]*[.]json\z/;
    exit 1 if $name =~ m{/};
    exit 1 if $name =~ /[[:cntrl:]]/;
    exit 1 if $seen{$name}++;
  }

  for my $entry (@entries) {
    my $path = "$output/" . $entry->full_path;
    sysopen(my $file, $path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600)
      or exit 1;
    binmode($file) or exit 1;
    print {$file} $entry->get_content or exit 1;
    close($file) or exit 1;
  }
' "${bundle}" "${output}" "${expected_count}" >/dev/null 2>&1; then
  fail
fi

if ! /usr/bin/perl -MFcntl=:mode -e '
  use strict;
  use warnings;

  my ($output, $expected_count) = @ARGV;
  my @directory_metadata = lstat($output);
  exit 1 unless @directory_metadata;
  exit 1 unless S_ISDIR($directory_metadata[2]);
  exit 1 unless ($directory_metadata[2] & 07777) == 0700;

  opendir(my $directory, $output) or exit 1;
  my @names = grep { $_ ne "." && $_ ne ".." } readdir($directory);
  closedir($directory) or exit 1;
  exit 1 unless @names == $expected_count;

  my %seen_names;
  my %seen_files;
  for my $name (@names) {
    exit 1 unless $name =~ /\A[A-Za-z0-9][A-Za-z0-9_-]*[.]json\z/;
    exit 1 if $name =~ m{/};
    exit 1 if $name =~ /[[:cntrl:]]/;
    exit 1 if $seen_names{$name}++;

    my @metadata = lstat("$output/$name");
    exit 1 unless @metadata;
    exit 1 unless S_ISREG($metadata[2]);
    exit 1 unless ($metadata[2] & 07777) == 0600;
    exit 1 unless $metadata[3] == 1;
    my $identity = "$metadata[0]:$metadata[1]";
    exit 1 if $seen_files{$identity}++;
  }
' "${output}" "${expected_count}" >/dev/null 2>&1; then
  fail
fi

output_owned=0
trap - EXIT HUP INT TERM
