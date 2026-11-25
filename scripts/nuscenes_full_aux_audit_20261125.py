"""Run the separate incremental shared-scan auxiliary audit."""
import runpy

if __name__ == '__main__':
    runpy.run_module('conceptgraph.completion.nuscenes_full_aux_audit_20261125', run_name='__main__')
