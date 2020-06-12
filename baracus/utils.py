import os
import subprocess
from subprocess import Popen, PIPE
import pandas as pd
import numpy as np
import statsmodels.api as sm
import nibabel as nb


def run(command, env={}, ignore_errors=False):
    merged_env = os.environ
    merged_env.update(env)
    # DEBUG env triggers freesurfer to produce gigabytes of files
    merged_env.pop('DEBUG', None)
    process = Popen(command, stdout=PIPE, stderr=subprocess.STDOUT, shell=True, env=merged_env)
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() != None:
            break
    if process.returncode != 0 and not ignore_errors:
        raise Exception("Non zero return code: %d" % process.returncode)


def run_fs_if_not_available(bids_dir, freesurfer_dir, subject_label, license_key, n_cpus, sessions=[], skip_missing=False):
    freesurfer_subjects = []
    if sessions:
        # long
        for session_label in sessions:
            freesurfer_subjects.append("sub-{sub}_ses-{ses}".format(sub=subject_label, ses=session_label))
    else:
        # cross
        freesurfer_subjects.append("sub-{sub}".format(sub=subject_label))

    fs_missing = False
    for fss in freesurfer_subjects:
        if not os.path.exists(os.path.join(freesurfer_dir, fss, "scripts/recon-all.done")):
            if skip_missing:
                freesurfer_subjects.remove(fss)
            else:
                fs_missing = True

    if fs_missing:
        cmd = "run_freesurfer.py {in_dir} {out_dir} participant " \
              "--hires_mode disable " \
              "--participant_label {subject_label} " \
              "--license_key {license_key} " \
              "--n_cpus {n_cpus} --steps cross-sectional".format(in_dir=bids_dir,
                                                                 out_dir=freesurfer_dir,
                                                                 subject_label=subject_label,
                                                                 license_key=license_key,
                                                                 n_cpus=n_cpus)

        print("Freesurfer for {} not found. Running recon-all: {}".format(subject_label, cmd))
        run(cmd)

    for fss in freesurfer_subjects:
        aseg_file = os.path.join(freesurfer_dir, fss, "stats/aseg.stats")
        if not os.path.isfile(aseg_file):
            if skip_missing:
                freesurfer_subjects.remove(fss)
            else:
                raise FileNotFoundError(aseg_file)
    return freesurfer_subjects


def get_subjects_session(layout, participant_label, truly_longitudinal_study):
    valid_subjects = layout.get_subjects(modality="anat", type="T1w")
    freesurfer_subjects = []

    if participant_label:
        subjects_to_analyze = set(participant_label) & set(valid_subjects)
        subjects_not_found = set(participant_label) - set(subjects_to_analyze)

        if subjects_not_found:
            raise Exception("Requested subjects not found or do not have required data: {}".format(subjects_not_found))
    else:
        subjects_to_analyze = valid_subjects

    sessions_to_analyze = {}
    for subject in subjects_to_analyze:
        if truly_longitudinal_study:
            sessions = layout.get_sessions(modality="anat", type="T1w", subject=subject)
            sessions_to_analyze[subject] = sessions
            for session in sessions:
                freesurfer_subjects.append("sub-{sub}_ses-{ses}".format(sub=subject, ses=session))
        else:
            freesurfer_subjects.append("sub-{sub}".format(sub=subject))

    return subjects_to_analyze, sessions_to_analyze, freesurfer_subjects



def get_residuals(X, Y):
    if len(Y.shape) == 1:
        Y = np.atleast_2d(Y).T
    betah = np.linalg.pinv(X).dot(Y)
    Yfitted = X.dot(betah)
    resid = Y - Yfitted
    return np.squeeze(betah[0] + resid.values)
    

def remove_confounds(data_files, confound_file):
    data_df = pd.DataFrame.from_dict(data_files, orient='index')
    confounds = pd.read_csv(confound_file)
    confounds = confounds.set_index(confounds.columns[0])
    if  (confounds.index.isin(data_df.index)).all():
        confounds = confounds.reindex(data_df.index)    
    else:
        raise Exception("Subjects in confound file and subject directory do not match. Make sure subject ID is in first column of confound file.")
    X = confounds
    X = sm.add_constant(X)
    all_surfs = ['lh_thickness_file', 'rh_thickness_file', 'lh_area_file', 'rh_area_file']
    for surf in all_surfs:
        filelist = data_df[surf]
        allsub_surf = []
        for f in filelist:
            img = nb.load(f)
            in_data = img.get_fdata().squeeze()
            allsub_surf.append(in_data)
        allsub_surf = pd.DataFrame(allsub_surf)
        surf_resid = allsub_surf.transform(lambda y: get_residuals(X,y), axis=0)
        for i, f in enumerate(filelist):
            out_data = surf_resid.iloc[i,:].values
            outimg = nb.freesurfer.mghformat.MGHImage(out_data.astype(np.float32), np.eye(4))
            nb.save(outimg, f)
            
    aseg_files = data_df['aseg_file']
    allsub_aseg = pd.DataFrame()
    for aseg_f in aseg_files:
        aseg_df = pd.read_csv(aseg_f, index_col=0, delimiter='\t')
        allsub_aseg = allsub_aseg.append(aseg_df)
    aseg_resid = allsub_aseg.transform(lambda y: get_residuals(X,y), axis=0)
    for i, f in enumerate(aseg_files):
        out_data = aseg_resid.iloc[[i]]
        out_data.to_csv(f, sep='\t', index=True)

    
