# Generating a mock project from the real one on Constructor Research Platform
The scripts and workflows are already on the platform. This section is for reference only.
1. Duplicate the "Sparse representation for machine learning the properties of defects in 2D materials" project
2. Install the [Chromium addon](https://github.com/kazeevn/mock-rolos-workflows/) for mocking workflows
3. Open the workflow page of the duplicated project, run the addon. *Don't* check the the "Overwrite the scripts" box.
4. Add `ai4material_design/scripts/Rolos/dry-run` to your project. Either through UI, or with terminal:
```bash
touch ai4material_design/scripts/Rolos/dry-run
git add ai4material_design/scripts/Rolos/dry-run
git commit -m "Add dry-run"
git push
```
5. Update the README to the mock readme, update the relative links. Use `CONSTRUCTOR-ALL.md` or `CONSTRUCTOR-MOCK.md`:
```bash
cp ai4material_design/docs/CONSTRUCTOR-ALL.md README.md
sed -i -E 's|(\[[^]]+]\()([^/h)][^)]+\))|\1ai4material_design/docs/\2|g' README.md 
git add README.md
git commit -m "Switch README to CONSTRUCTOR-MOCK"
git push
```